from file_extractor import extract_resume_text
from text_preprocessing import clean_text
from skill_extractor import extract_skills_with_spacy

def main():
    # Replace with your resume file path
    file_path = "../data/sample_resumes/resume1.pdf"
    
    # 1. Extract text
    raw_text = extract_resume_text(file_path)
    if not raw_text:
        print("Failed to extract text from the resume.")
        return

    # 2. Preprocess the text
    cleaned_text = clean_text(raw_text)
    
    # 3. Extract skills
    skills_spacy = extract_skills_with_spacy(cleaned_text)
    
    print("Extracted Skills (spaCy NER):", skills_spacy)

if __name__ == "__main__":
    main()
