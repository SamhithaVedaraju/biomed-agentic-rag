from biorag.data.documents import Document, chunk_document


def test_short_document_is_one_chunk():
    doc = Document(doc_id="1", title="T", text="One sentence. Two sentences.")
    chunks = chunk_document(doc, size=1000)
    assert [c.chunk_id for c in chunks] == ["1#0"]
    assert chunks[0].render().startswith("T\n")


def test_long_document_splits_with_overlap():
    sentences = [f"Sentence number {i} is here." for i in range(40)]
    doc = Document(doc_id="9", text=" ".join(sentences))
    chunks = chunk_document(doc, size=200, overlap=60)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 + 30 for c in chunks)
    # The last sentence of one chunk reappears at the start of the next.
    first_tail = chunks[0].text.split(". ")[-1]
    assert first_tail.rstrip(".") in chunks[1].text
    # Nothing is lost.
    joined = " ".join(c.text for c in chunks)
    assert all(s in joined for s in sentences)
