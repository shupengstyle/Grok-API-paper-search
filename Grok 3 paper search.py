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
from openai import OpenAI
import google.generativeai as genai
import hashlib
import sqlite3
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables from .env file
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.yeah.net")
SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 465))
SEARCH_QUERY = os.getenv("SEARCH_QUERY")
MAX_RESULTS = int(os.getenv("MAX_RESULTS", 15))
XAI_API_KEY = os.getenv("XAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUMMARY_LANGUAGE = os.getenv("SUMMARY_LANGUAGE", "en")

# Check XAI_API_KEY
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY 未在环境变量中设置，请检查 .env 文件")
logging.info(f"XAI_API_KEY 已加载: {XAI_API_KEY[:4]}****")

# Initialize Entrez
Entrez.email = EMAIL_ADDRESS
Entrez.api_key = PUBMED_API_KEY

# Initialize Grok API client
client = OpenAI(api_key=XAI_API_KEY, base_url="https://api.x.ai/v1")

# Initialize Gemini API client
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

# Database configuration
DATABASE_FILE = "processed_articles.db"
DATA_RETENTION_DAYS = 30


# Grok API call with retry
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_grok_api(prompt):
    try:
        response = client.chat.completions.create(
            model="grok-2-latest",
            messages=[
                {"role": "system", "content": "You are Grok, a helpful AI specialized in processing scientific texts."},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content.strip() if response.choices else "Grok response unavailable"
    except Exception as e:
        logging.error(f"Grok API 调用失败: {str(e)}")
        raise


# Database functions (unchanged)
def generate_article_hash(article):
    if article.get("pmid") and article["pmid"] != "无PMID":
        identifier = article["pmid"]
    elif article.get("doi") and article["doi"] != "无DOI":
        identifier = article["doi"]
    else:
        return None
    return hashlib.md5(identifier.encode('utf-8')).hexdigest()


def create_connection():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        logging.info(f"✅ 成功连接到数据库: {DATABASE_FILE}")
    except sqlite3.Error as e:
        logging.error(f"❌ 数据库连接失败: {e}")
    return conn


def create_table(conn):
    try:
        sql = """
        CREATE TABLE IF NOT EXISTS processed_articles (
            article_hash TEXT PRIMARY KEY,
            pmid TEXT,
            doi TEXT,
            title TEXT,
            processed_date TEXT
        );
        """
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        logging.info("✅ 成功创建 processed_articles 表")
    except sqlite3.Error as e:
        logging.error(f"❌ 创建表失败: {e}")


def load_processed_articles(conn):
    processed_articles = {}
    try:
        sql = "SELECT article_hash, pmid, doi FROM processed_articles"
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            article_hash, pmid, doi = row
            processed_articles[article_hash] = {"pmid": pmid, "doi": doi}
        logging.info(f"✅ 加载了 {len(processed_articles)} 篇文章")
    except sqlite3.Error as e:
        logging.error(f"❌ 从数据库加载数据失败: {e}")
    return processed_articles


def save_processed_article(conn, article_hash, pmid, doi, title):
    try:
        sql = "INSERT INTO processed_articles (article_hash, pmid, doi, title, processed_date) VALUES (?, ?, ?, ?, ?)"
        cursor = conn.cursor()
        cursor.execute(sql, (article_hash, pmid, doi, title, datetime.now().isoformat()))
        conn.commit()
        logging.info(f"✅ 保存文章到数据库: PMID {pmid}, DOI: {doi}, 标题: {title}")
    except sqlite3.Error as e:
        logging.error(f"❌ 保存文章到数据库失败: {e}")


def delete_expired_articles(conn, retention_days=DATA_RETENTION_DAYS):
    try:
        expiry_date = datetime.now() - timedelta(days=retention_days)
        expiry_date_str = expiry_date.isoformat()
        sql = "DELETE FROM processed_articles WHERE processed_date < ?"
        cursor = conn.cursor()
        cursor.execute(sql, (expiry_date_str,))
        deleted_count = cursor.rowcount
        conn.commit()
        logging.info(f"✅ 成功删除 {deleted_count} 篇超过 {retention_days} 天的过期文章")
    except sqlite3.Error as e:
        logging.error(f"❌ 删除过期文章失败: {e}")


# Fetch articles (unchanged)
def fetch_articles():
    try:
        handle = Entrez.esearch(db="pubmed", term=SEARCH_QUERY, retmax=MAX_RESULTS, sort="pub_date")
        result = Entrez.read(handle)
        id_list = result["IdList"]
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
                doi = article.find("ELocationID", {"EIdType": "doi"}).get_text() if article.find("ELocationID", {
                    "EIdType": "doi"}) else "无DOI"
                pmcid = article.find("ArticleId", {"IdType": "pmc"}).get_text() if article.find("ArticleId", {
                    "IdType": "pmc"}) else "无PMCID"
                authors = [f"{author.find('LastName').get_text()} {author.find('Initials').get_text()}"
                           for author in article.find_all("Author")
                           if author.find("LastName") and author.find("Initials")]
                abstract = "\n".join(
                    [text.get_text() for text in article.find("Abstract").find_all("AbstractText")]) if article.find(
                    "Abstract") else ""
                articles.append(
                    {"pmid": pmid, "title": title, "journal": journal, "year": year, "doi": doi, "pmcid": pmcid,
                     "authors": authors, "abstract": abstract})
            except Exception as e:
                logging.error(f"解析PMID {pmid} 时发生错误: {e}")
                continue
        return articles
    except Exception as e:
        logging.error(f"文献搜索失败: {str(e)}")
        return []


# Async full-text fetching
async def fetch_fulltext(session, url):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            return await response.text() if response.status == 200 else None
    except Exception as e:
        logging.error(f"异步请求失败: {str(e)}，URL: {url}")
        return None


async def get_fulltexts(articles):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for article in articles:
            if article["doi"] != "无DOI":
                tasks.append(fetch_fulltext(session, f"https://doi.org/{article['doi']}"))
            elif article["pmcid"] != "无PMCID":
                tasks.append(fetch_fulltext(session,
                                            f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{article['pmcid']}/?format=txt"))
            else:
                tasks.append(asyncio.ensure_future(asyncio.sleep(0, result=None)))
        return await asyncio.gather(*tasks)


# Translation and summarization
def translate_text(text, target_language="zh-CN"):
    if not text:
        return ""
    try:
        prompt = f"Translate the following text to {target_language}: {text}"
        response = call_grok_api(prompt)
        return response.strip()
    except Exception as e:
        logging.error(f"Grok翻译失败: {str(e)}, 尝试使用Gemini")
        return translate_text_gemini(text, target_language)


def summarize_text(text, target_language="en"):
    if not text:
        return "无内容可总结"
    try:
        prompt = f"""
        Please provide an academic summary of the following medical research article,
        ensuring it encompasses the study's background, the methodology used,
        the principal research results obtained, and an assessment of the research's significance and value.
        Text: {text}
        """
        response = call_grok_api(prompt)
        cleaned_text = response.replace("**", "").replace("*", "").replace("■", "").replace("●", "").replace("◆", "")
        return cleaned_text.strip()
    except Exception as e:
        logging.error(f"Grok总结失败: {str(e)}, 尝试使用Gemini")
        return summarize_text_gemini(text, target_language)


# Gemini fallback functions (unchanged)
def translate_text_gemini(text, target_language="zh-CN"):
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
    if not text:
        return "无内容可总结"
    try:
        prompt = f"""
        Please provide an academic summary of the following medical research article,
        ensuring it encompasses the study's background, the methodology used,
        the principal research results obtained, and an assessment of the research's significance and value.
        Text: {text}
        """
        response = gemini_model.generate_content(prompt)
        return response.text.strip() if response.text else "无法生成总结"
    except Exception as e:
        logging.error(f"Gemini API 总结失败: {str(e)}")
        return "无法生成总结"


# Email function with abstract added
def send_email(articles, processed_articles):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = EMAIL_ADDRESS
        msg["Subject"] = f"PubMed文献更新 - 髋膝关节置换文献"
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
                <p><b>英文摘要:</b><br>{article['abstract']}</p>
                <p><b>中文总结:</b><br>{article['summary']}</p>
                <p>
                    <a href="{article['link']}">PubMed链接</a> |
                    <a href="https://doi.org/{article['doi']}">全文链接</a>
                </p>
            </div>
            """
        html_content += "</body></html>"
        msg.attach(MIMEText(html_content, "html", "utf-8"))
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            logging.info("✅ 邮件发送成功")
    except Exception as e:
        logging.error(f"❌ 邮件发送失败: {str(e)}")


# Main function with async full-text fetching
def main():
    logging.info("🚀 开始执行...")
    conn = create_connection()
    if conn is None:
        return
    create_table(conn)
    delete_expired_articles(conn)
    processed_articles = load_processed_articles(conn)
    logging.info("🔄 获取并总结文献...")
    all_articles = fetch_articles()
    if not all_articles:
        logging.warning("❌ 未找到相关文献")
        conn.close()
        return

    # Async fetch full texts
    fulltexts = asyncio.run(get_fulltexts(all_articles))

    new_articles = []
    for idx, article in enumerate(all_articles):
        article_hash = generate_article_hash(article)
        if article_hash is None:
            logging.warning(f"⚠️ 没有PMID和DOI, 无法判断重复性, 标题: {article['title']}")
            continue
        if article_hash not in processed_articles:
            fulltext = fulltexts[idx]
            summary = summarize_text(fulltext if fulltext else article["abstract"] or "无摘要")
            translated_summary = translate_text(summary, target_language="zh-CN")
            translated_title = translate_text(article["title"], target_language="zh-CN")
            article["summary"] = translated_summary
            article["translated_title"] = translated_title
            article["link"] = f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/"
            new_articles.append(article)
            save_processed_article(conn, article_hash, article['pmid'], article['doi'], article['title'])
            processed_articles[article_hash] = {"pmid": article['pmid'], "doi": article['doi']}
            logging.info(f"已添加新文章: PMID {article['pmid']}, DOI: {article['doi']}, 标题: {article['title']}")
            time.sleep(0.5)
        else:
            logging.info(f"文章已存在: PMID {article['pmid']}, DOI: {article['doi']}, 标题: {article['title']}")

    logging.info("📧 发送邮件...")
    if new_articles:
        send_email(new_articles, processed_articles)
    else:
        logging.info("❌ 没有新的文献需要发送")
    conn.close()
    logging.info("✅ 数据库连接已关闭")


if __name__ == "__main__":
    main()
