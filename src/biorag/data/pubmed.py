"""Fetch abstracts from PubMed via NCBI E-utilities.

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
NCBI asks for a contact email on every request and allows 3 req/s (10 with an API key).
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator

import httpx

from biorag.data.documents import Document

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
BATCH = 200


def search_pmids(client: httpx.Client, term: str, retmax: int, params: dict) -> list[str]:
    r = client.get(
        f"{EUTILS}/esearch.fcgi",
        params={**params, "db": "pubmed", "term": term, "retmax": retmax, "retmode": "json"},
    )
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def parse_articles(xml_text: str) -> Iterator[Document]:
    root = ET.fromstring(xml_text)
    for article in root.iter("PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title_node = article.find(".//ArticleTitle")
        title = "".join(title_node.itertext()) if title_node is not None else ""
        sections = []
        for node in article.findall(".//Abstract/AbstractText"):
            text = "".join(node.itertext()).strip()
            label = node.get("Label")
            if text:
                sections.append(f"{label.title()}: {text}" if label else text)
        if pmid and sections:
            yield Document(doc_id=pmid, title=title.strip(), text=" ".join(sections))


def fetch_abstracts(
    term: str | None,
    retmax: int,
    email: str,
    api_key: str | None = None,
    pmids: list[str] | None = None,
) -> Iterator[Document]:
    """Fetch abstracts for a search `term`, or for an explicit list of `pmids`."""
    params = {"tool": "biorag", "email": email}
    if api_key:
        params["api_key"] = api_key
    delay = 0.11 if api_key else 0.34

    with httpx.Client(timeout=60) as client:
        if pmids is None:
            pmids = search_pmids(client, term, retmax, params)
        for start in range(0, len(pmids), BATCH):
            batch = pmids[start : start + BATCH]
            time.sleep(delay)
            r = client.post(
                f"{EUTILS}/efetch.fcgi",
                data={**params, "db": "pubmed", "id": ",".join(batch), "retmode": "xml"},
            )
            r.raise_for_status()
            yield from parse_articles(r.text)
