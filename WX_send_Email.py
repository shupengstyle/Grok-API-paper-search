import os
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from Bio import Entrez
from bs4 import BeautifulSoup
import requests
import logging
import dotenv
import json
from datetime import datetime, timedelta
import openai
import google.generativeai as genai  # 导入 Gemini API

# Load environment variables from .env file
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 环境变量配置
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.yeah.net")
SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 465))
SEARCH_QUERY = os.getenv("SEARCH_QUERY")
MAX_RESULTS = int(os.getenv("MAX_RESULTS", 5))
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # 新增: Gemini API Key
SUMMARY_LANGUAGE = os.getenv("SUMMARY_LANGUAGE", "en")
PROCESSED_PMIDS_FILE = os.getenv("PROCESSED_PMIDS_FILE", "processed_pmids.json")
PROCESSED_PMIDS_EXPIRATION_DAYS = int(os.getenv("PROCESSED_PMIDS_EXPIRATION_DAYS", 30))

# 初始化配置
Entrez.email = EMAIL_ADDRESS
Entrez.api_key = PUBMED_API_KEY

# 初始化 DeepSeek API 客户端
client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 初始化 Gemini API 客户端
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')  # 选择合适的 Gemini 模型


def load_processed_pmids():
    """Load processed PMIDs from file."""
    try:
        with open(PROCESSED_PMIDS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:  # Handle corrupted JSON file
        logging.warning("Processed PMIDs file is corrupted. Starting with an empty list.")
        return []


def save_processed_pmids(pmids):
    """Save processed PMIDs to file."""
    try:
        with open(PROCESSED_PMIDS_FILE, "w") as f:
            json.dump(pmids, f)
    except Exception as e:
        logging.error(f"Failed to save processed PMIDs to file: {e}")


def fetch_articles():
    """获取文献列表和基本信息"""
    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=SEARCH_QUERY,
            retmax=MAX_RESULTS,
            sort="pub_date",
            usehistory="y"
        )
        result = Entrez.read(handle)
        id_list = result["IdList"]

        # Fetch article details in batch
        handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="xml", retmode="text")
        xml_data = handle.read()
        soup = BeautifulSoup(xml_data, "lxml-xml")

        articles = []
        for article in soup.find_all("PubmedArticle"):
            try:
                pmid = article.find("PMID").get_text()
                title = article.find("ArticleTitle").get_text() if article.find("ArticleTitle") else "无标题"
                journal = article.find("ISOAbbreviation").get_text() if article.find("ISOAbbreviation") else "未知杂志"
                year = article.find("Year").get_text() if article.find("Year") else "未知年份"
                doi = article.find("ELocationID", {"EIdType": "doi"}).get_text() if article.find("ELocationID", {"EIdType": "doi"}) else "无DOI"
                pmcid = article.find("ArticleId", {"IdType": "pmc"}).get_text() if article.find("ArticleId", {"IdType": "pmc"}) else "无PMCID"

                # 处理作者信息
                authors = []
                for author in article.find_all("Author"):
                    lastname = author.find("LastName").get_text() if author.find("LastName") else ""
                    initials = author.find("Initials").get_text() if author.find("Initials") else ""
                    if lastname and initials:
                        authors.append(f"{lastname} {initials}")

                # 处理摘要
                abstract = ""
                if abstract_section := article.find("Abstract"):
                    abstract = "\n".join([text.get_text() for text in abstract_section.find_all("AbstractText")])

                articles.append({
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "year": year,
                    "doi": doi,
                    "pmcid": pmcid,
                    "authors": authors,
                    "abstract": abstract
                })
            except Exception as e:
                logging.error(f"解析PMID {pmid} 时发生错误: {e}")
                continue  # 继续处理下一个文章

        return articles

    except Exception as e:
        logging.error(f"文献搜索失败: {str(e)}")
        return []


def get_fulltext_by_doi(doi):
    """尝试通过DOI获取全文"""
    if doi == "无DOI":
        return None

    try:
        url = f"https://doi.org/{doi}"
        response = requests.get(url, headers={"Accept": "text/plain"}, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            print(f"通过DOI获取全文失败 (Status Code: {response.status_code}), doi: {doi}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"请求DOI全文失败: {str(e)}, doi: {doi}")
        return None


def get_fulltext_by_pmcid(pmcid):
    """尝试通过PMCID获取全文"""
    if not pmcid or pmcid == "无PMCID":
        return None

    try:
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/?format=txt"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            print(f"通过PMCID获取全文失败 (Status Code: {response.status_code}), pmcid: {pmcid}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"请求PMCID全文失败: {str(e)}, pmcid: {pmcid}")
        return None


def translate_text(text, target_language="zh-CN"):
    """使用 DeepSeek API 翻译文本, 失败则使用 Gemini API"""
    if not text:
        return ""
    try:
        prompt = f"Translate the following text to {target_language}: {text}"
        response = client.chat.completions.create(
            model="deepseek-chat",  # 或者选择其他适合翻译的模型
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in translating scientific articles."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content.strip() if response.choices else text
    except Exception as e:
        logging.error(f"DeepSeek API 翻译失败: {str(e)}, 尝试使用 Gemini API")
        return translate_text_gemini(text, target_language)  # 调用 Gemini API


def summarize_text(text, target_language="en"):
    """使用 DeepSeek API 生成文本总结, 失败则使用 Gemini API"""
    if not text:
        return "无内容可总结"
    try:
        prompt = f"""
        Please provide an academic summary of the following medical research article, 
        ensuring it encompasses the study's background, the methodology used, 
        the principal research results obtained, and an assessment of the research's significance and value. 
        The summary should be clear, concise, and free of unnecessary detail. 
        文本：
        {text}
        """
        response = client.chat.completions.create(
            model="deepseek-chat",  # 或者选择其他适合摘要的模型
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in summarizing scientific articles."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        # 清理 DeepSeek API 返回的文本
        summary = response.choices[0].message.content.strip() if response.choices else "无法生成总结"
        cleaned_text = summary.replace("**", "").replace("*", "", ).replace("■", "").replace("●", "").replace("◆", "")
        return cleaned_text
    except Exception as e:
        logging.error(f"DeepSeek API 总结失败: {str(e)}, 尝试使用 Gemini API")
        return summarize_text_gemini(text, target_language)  # 调用 Gemini API


def translate_text_gemini(text, target_language="zh-CN"):
    """使用 Gemini API 翻译文本"""
    if not text:
        return ""
    try:
        prompt = f"Translate the following text to {target_language}: {text}"
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response.text else text
    except Exception as e:
        logging.error(f"Gemini API 翻译失败: {str(e)}")
        return text


def summarize_text_gemini(text, target_language="en"):
    """使用 Gemini API 生成文本总结"""
    if not text:
        return "无内容可总结"
    try:
        prompt = f"""
        Please provide an academic summary of the following medical research article, 
        ensuring it encompasses the study's background, the methodology used, 
        the principal research results obtained, and an assessment of the research's significance and value. 
        The summary should be clear, concise, and free of unnecessary detail. 
        文本：
        {text}
        """
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response.text else "无法生成总结"
    except Exception as e:
        logging.error(f"Gemini API 总结失败: {str(e)}")
        return "无法生成总结"


def cleanup_processed_pmids(pmids):
    """Cleanup processed PMIDs, removing entries older than PROCESSED_PMIDS_EXPIRATION_DAYS."""
    cutoff_date = datetime.now() - timedelta(days=PROCESSED_PMIDS_EXPIRATION_DAYS)
    cleaned_pmids = []
    for entry in pmids:
        try:
            # Assuming the timestamp is stored as a string in ISO format
            entry_date = datetime.fromisoformat(entry["timestamp"])
            if entry_date >= cutoff_date:
                cleaned_pmids.append(entry)
        except (KeyError, ValueError) as e:
            logging.warning(f"Invalid entry format in processed PMIDs: {entry}. Error: {e}. Skipping.")
            continue  # Skip invalid entries
    logging.info(f"Cleaned up processed PMIDs, removed {len(pmids) - len(cleaned_pmids)} entries.")
    return cleaned_pmids


def send_email(articles):
    """发送文献汇总邮件"""
    try:
        # 构建邮件内容
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = EMAIL_ADDRESS
        msg["Subject"] = f"PubMed文献更新 - {SEARCH_QUERY}"

        # HTML内容模板
        html_content = f"""
        <html>
            <body>
                <h2>最新 {len(articles)} 篇文献</h2>
                <p>搜索关键词: {SEARCH_QUERY}</p>
        """

        for idx, article in enumerate(articles, 1):
            html_content += f"""
            <div style="margin-bottom: 30px; border-bottom: 1px solid #eee;">
                <h3>{idx}. {article['title']}</h3>
                 <p><b>中文标题:</b> {article['translated_title']}</p>
                <p><b>作者:</b> {', '.join(article['authors'])}</p>
                <p><b>期刊:</b> {article['journal']} ({article['year']})</p>
                <p><b>中文总结:</b><br>{article['summary']}</p>
                <p>
                    <a href="{article['link']}">PubMed链接</a> |
                    <a href="https://doi.org/{article['doi']}">全文链接</a>
                </p>
            </div>
            """

        html_content += "</body></html>"
        msg.attach(MIMEText(html_content, "html", "utf-8"))

        # 发送邮件
        try:
            logging.info(f"Connecting to SMTP server: {SMTP_SERVER}:{SMTP_PORT}")
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
                logging.info(f"Logging in to email account: {EMAIL_ADDRESS}")
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                logging.info("Login successful.")
                logging.info(f"Sending email to: {EMAIL_ADDRESS}")
                server.send_message(msg)
                logging.info("✅ 邮件发送成功")
        except smtplib.SMTPException as e:
            logging.error(f"❌ SMTP error occurred: {e}")
            raise  # Re-raise the exception to be caught in the outer block

    except Exception as e:
        logging.error(f"❌ 邮件发送失败: {str(e)}")


def is_processed(processed_pmids, article):
    """检查文章是否已被处理，优先使用PMCID，如果PMCID不存在则使用DOI"""
    for entry in processed_pmids:
        if article["pmcid"] != "无PMCID" and entry.get("pmcid") == article["pmcid"]:  # 优先匹配PMCID
            return True
        elif article["doi"] != "无DOI" and entry.get("doi") == article["doi"]:  # 如果没有PMCID，则匹配DOI
            return True
    return False


if __name__ == "__main__":
    logging.info("🚀 开始获取文献...")
    all_articles = fetch_articles()

    if not all_articles:
        logging.warning("❌ 未找到相关文献")
        exit()

    processed_pmids = load_processed_pmids()
    processed_pmids = cleanup_processed_pmids(processed_pmids)

    new_articles = []

    for article in all_articles:
        if not is_processed(processed_pmids, article):
            # 获取全文
            fulltext = get_fulltext_by_doi(article["doi"])
            if not fulltext:
                fulltext = get_fulltext_by_pmcid(article["pmcid"])

            # 总结和翻译
            if fulltext:
                summary = summarize_text(fulltext)
            else:
                summary = summarize_text(article["abstract"] or "无摘要")

            translated_summary = translate_text(summary, target_language="zh-CN")
            translated_title = translate_text(article["title"], target_language="zh-CN")

            # 将结果添加到 new_articles
            article["summary"] = translated_summary
            article["translated_title"] = translated_title
            article["link"] = f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/"
            new_articles.append(article)

            # 保存已处理的 PMID 信息
            processed_pmids.append({
                "pmid": article["pmid"],
                "timestamp": datetime.now().isoformat(),
                "pmcid": article["pmcid"],
                "doi": article["doi"]
            })
            logging.info(f"已添加新文章: PMID {article['pmid']}, 标题: {article['title']}")
            time.sleep(0.5)  # 避免速率限制

        else:
            logging.info(f"PMID {article['pmid']} (PMCID: {article['pmcid']}, DOI: {article['doi']}) 已经处理过，跳过")

    if new_articles:
        send_email(new_articles)  # 只发送新的文献
        save_processed_pmids(processed_pmids)  # 保存已处理的PMID列表
    else:
        logging.info("❌ 没有新的文献需要发送")
