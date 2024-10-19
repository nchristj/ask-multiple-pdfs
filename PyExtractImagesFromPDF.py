import fitz
import io
from PIL import Image

def extract_images_from_pdf(pdf_file):
    # Open the PDF
    pdf_document = fitz.open(pdf_file)
    image_count = 0

    # Iterate through the pages
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)  # Load the page
        image_list = page.get_images(full=True)  # Get all images from the page

        # If there are images, extract them
        for image_index, img in enumerate(image_list):
            xref = img[0]  # Reference number for the image
            base_image = pdf_document.extract_image(xref)  # Extract image using the xref number
            image_bytes = base_image["image"]

            # Get the image extension (e.g., PNG, JPEG)
            image_ext = base_image["ext"]

            # Convert the byte data to a PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Save the image with a unique name
            image_filename = f"ExtractedImage/extracted_image_{page_num + 1}_{image_index + 1}.{image_ext}"
            image.save(image_filename)

            image_count += 1
            print(f"Saved: {image_filename}")

    print(f"Total images extracted: {image_count}")
    pdf_document.close()

# Example usage
pdf_file = "newassetallocationreport.pdf"  # Path to your PDF file
extract_images_from_pdf(pdf_file)
