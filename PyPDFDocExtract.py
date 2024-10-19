import fitz

pdf_document = fitz.open(filename='C:/Users/chris/OneDrive/Desktop/Functional_Requirement_Document_2.pdf', filetype="pdf")

headers = []
# Iterate through each page in the document
chunks = []
current_chunk = {"header": None, "content": ""}

# Iterate through each page in the document
for page_num, page in enumerate(pdf_document, start=1):
    blocks = page.get_text("dict")["blocks"]

    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    font_size = span["size"]  # Font size of the text
                    text = span["text"].strip()

                    # If the font size is larger than a threshold, treat it as a header
                    if font_size > 12 and text:
                        print(font_size)
                        print(text)
                        # If there's an existing header, save its chunk
                        if current_chunk["header"]:
                            chunks.append(current_chunk)

                        # Start a new chunk for the new header
                        current_chunk = {"header": text, "content": ""}
                    else:
                        # Append the text to the current header's content
                        current_chunk["content"] += text + " "

# Add the last chunk if there's remaining content
if current_chunk["header"]:
    chunks.append(current_chunk)

result_chunks = []
# Print the header and its content for each chunk
for chunk in chunks:
    print(f"Header: {chunk['header']}")
    #print(f"Content: {chunk['content']}\n")
    text = chunk['header']+"\n"+chunk['content']
    print(len(text))
    result_chunks.append(text)
print(len(result_chunks))
print(result_chunks)


def extract_header_content_chunks_with_distance(file):
    # Open the PDF document
    pdf_document = fitz.open(filename=file, filetype="pdf")

    chunks = []
    current_chunk = {"header": None, "content": ""}
    previous_bottom = None
    threshold_font_size = 12  # Adjust font size for headers
    threshold_distance = 20  # Adjust vertical distance threshold (in points)

    # Iterate through each page in the document
    for page_num, page in enumerate(pdf_document, start=1):
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_size = span["size"]  # Font size of the text
                        text = span["text"].strip()
                        bbox = span["bbox"]  # Bounding box (x0, y0, x1, y1)

                        current_top = bbox[1]  # The top of the current text block
                        current_bottom = bbox[3]  # The bottom of the current text block

                        # Calculate the vertical distance between current and previous line
                        if previous_bottom is not None:
                            vertical_distance = current_top - previous_bottom
                        else:
                            vertical_distance = 0  # No distance for the first line

                        # Determine if this is a header based on font size or large vertical gap
                        if font_size > threshold_font_size or vertical_distance > threshold_distance:
                            # If there's an existing header, save its chunk
                            if current_chunk["header"]:
                                chunks.append(current_chunk)

                            # Start a new chunk for the new header
                            current_chunk = {"header": text, "content": ""}
                        else:
                            # Append the text to the current header's content
                            current_chunk["content"] += text + " "

                        # Update previous bottom for the next iteration
                        previous_bottom = current_bottom

    # Add the last chunk if there's remaining content
    if current_chunk["header"]:
        chunks.append(current_chunk)

    # Print the header and its content for each chunk
    for chunk in chunks:
        print(f"Header: {chunk['header']}")
        print(f"Content: {chunk['content']}\n")
        print(len(chunk['content']))

    return chunks

extract_header_content_chunks_with_distance("C:/Users/chris/OneDrive/Desktop/Functional_Requirement_Document_2.pdf")