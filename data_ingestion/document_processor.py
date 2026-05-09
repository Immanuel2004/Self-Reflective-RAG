import re
import fitz
from typing import List, Dict

def ingest_bible_pdf(pdf_path: str, source_name: str, verse_window: int = 6, overlap: int = 0) -> List[Dict]:
    doc = fitz.open(pdf_path)
    all_verses = []
    current_book = "Genesis" # Default to start
    current_chapter = 1
    found_start = False 

    BOOKS = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth",
             "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra",
             "Nehemiah", "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon",
             "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
             "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah",
             "Malachi", "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians",
             "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians",
             "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon",
             "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation"]

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        for b in blocks:
            if "lines" not in b: continue
            
            block_text = ""
            for line in b["lines"]:
                for span in line["spans"]:
                    if span["size"] < 7: continue 
                    block_text += span["text"] + " "
            
            text = block_text.strip()
            if not text: continue

            if not found_start:
                if "In the beginning" in text:
                    found_start = True
                else: continue


            first_word = text.split(' ')[0].title()
            first_two = " ".join(text.split(' ')[:2]).title()
            
            if first_word in BOOKS:
                current_book = first_word
                current_chapter = 1
            elif first_two in BOOKS:
                current_book = first_two
                current_chapter = 1

            if re.match(r"^(Chapter\s+)?(\d+)$", text, re.IGNORECASE):
                current_chapter = int(re.search(r"\d+", text).group())
                continue

            parts = re.split(r'(\d+)\s*(?=[A-Z])', text)
            if len(parts) > 1:
                for i in range(1, len(parts), 2):
                    v_num = int(parts[i])
                    v_raw = parts[i+1].strip()
                    v_clean = re.sub(r'\[\d+\]', '', v_raw).strip() # Remove [ref]
                    
                    if 0 < v_num < 180 and len(v_clean) > 5:
                        all_verses.append({
                            "book": current_book,
                            "chapter": current_chapter,
                            "verse": v_num,
                            "text": v_clean
                        })

    doc.close()
    
    chunks = []
    if verse_window <= 0:
        raise ValueError("verse_window must be > 0")

    step = verse_window - overlap if (verse_window - overlap) > 0 else verse_window
    for i in range(0, len(all_verses), step):
        group = all_verses[i : i + verse_window]
        if not group:
            continue
        content = " ".join([f"{v['verse']} {v['text']}" for v in group])
        f, l = group[0], group[-1]
        if f['book'] == l['book'] and f['chapter'] == l['chapter']:
            ref = f"{f['book']} {f['chapter']}:{f['verse']}-{l['verse']}"
        else:
            # spanning chapter or book boundaries: simplify reference
            ref = f"{f['book']} {f['chapter']}:{f['verse']} - {l['book']} {l['chapter']}:{l['verse']}"

        chunks.append({
            "content": content,
            "metadata": {
                "book": f["book"],
                "chapter": f["chapter"],
                "reference": ref,
                "source": source_name,
                "verse_window": verse_window,
                "overlap": overlap,
            },
        })
    
    return chunks

def create_clean_pdf(input_path: str, output_path: str):
    """Utility to generate the clean PDF from boundaries."""
    src = fitz.open(input_path)
    dst = fitz.open()
    start, end = -1, -1
    
    for i in range(len(src)):
        txt = src[i].get_text().lower()
        if start == -1 and "in the beginning" in txt: start = i
        if "grace of the lord jesus be with all" in txt: end = i
            
    if start != -1:
        dst.insert_pdf(src, from_page=start, to_page=(end if end != -1 else len(src)-1))
        dst.save(output_path)
    src.close()
    dst.close()