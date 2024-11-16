import streamlit as st
from PIL import Image
import base64
from PyPDF2 import PdfReader
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from bs4 import BeautifulSoup
import pandas as pd

#note book : "https://wallpapers.com/images/featured/notebook-background-vo962ec6p7mb0jw4.jpg"
# https://img.freepik.com/free-vector/dark-polygonal-background_79603-282.jpg
#https://img.pikbest.com/wp/202344/crumpled-paper-texture-background-recycled-with-appearance_9923149.jpg!sw800
pageBase = """
<style>
[data-testid="stMain"]{
    
    //background-image: url("https://img.freepik.com/premium-photo/rough-kraft-paper-background-paper-texture-lilac-white-colors-mockup-with-copy-space-text_154092-26497.jpg?w=740");
    //background-image: url("https://www.shutterstock.com/image-photo/crumpled-light-purple-paper-texture-600nw-2217136101.jpg");
    background-image: url("https://kartinki.pics/uploads/posts/2021-07/1626156998_12-kartinkin-com-p-zelenii-velyur-tekstura-besshovnaya-krasiv-12.jpg");
    background-size: 100% auto;
    background-position: center center;
    }

[data-testid="stHeader"]{
   
    background-color: transparent;
   }

[data-testid="stTab"]{
   
    //background-color: #000000;
    color: #000000;
    font-weight: bold;
    font-size: 45pt;
    
   }

/* Container styling */
.mainContainer {
    background-image: url("https://static.vecteezy.com/system/resources/thumbnails/018/795/029/small_2x/white-notebook-paper-with-all-edges-ripped-png.png");
    background-size: cover;
    padding: 20pt 20pt 10pt 60pt;
    
    width: 100%; /* Make each container take up 45% of the wrapper width */
   // height: 100pt;
    text-align: center; /* Center-align text inside each container */
}


/* Heading style */
.mainContainer h1 {
    color: #333; /* Dark grey color for the heading */
    font-size: 30px; /* Adjust font size */
    margin-bottom: 5px; /* Space below the heading */
}

/* Paragraph style */
.mainContainer p {
    color: #555; /* Medium grey color for the text */
    font-size: 17px; /* Adjust font size */
}

/* extra styling */
.extra {
    background-image: url("2.png");
    background-size: auto;
    padding: 20pt 20pt 10pt 60pt;
    
    width: 100%; /* Make each container take up 45% of the wrapper width */
    min-height: 100%; /* Minimum height to show background image */
    text-align: center; /* Center-align text */
    
}


/* Heading style */
.extra h1 {
    color: #333; /* Dark grey color for the heading */
    font-size: 30px; /* Adjust font size */
    margin-bottom: 5px; /* Space below the heading */
}

/* Paragraph style */
.extra p {
    color: #555; /* Medium grey color for the text */
    font-size: 17px; /* Adjust font size */
}


/* Container styling */
.mainContainerGrade {
    background-image: url("https://static.vecteezy.com/system/resources/thumbnails/018/795/029/small_2x/white-notebook-paper-with-all-edges-ripped-png.png");
    background-size: cover;
    padding: 20pt 30pt 20pt 60pt;
    position: absolute;
    right: 15%;
    width: 70%; /* Make each container take up 45% of the wrapper width */
   // height: 100pt;
    text-align: center; /* Center-align text inside each container */
}


/* Heading style */
.mainContainerGrade h1 {
    color: #333; /* Dark grey color for the heading */
    font-size: 30px; /* Adjust font size */
    margin-bottom: 15px; /* Space below the heading */
}

/* Paragraph style */
.mainContainerGrade p {
    color: #555; /* Medium grey color for the text */
    font-size: 17px; /* Adjust font size */
}


/* Container styling */
.container {
    background-color: white; /* White background */
    padding: 10px; /* Padding inside the container */
    border: 1px solid #ddd; /* Light grey border */
    border-radius: 8px; /* Rounded corners */
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); /* Soft shadow */
    width: 100%; /* Make each container take up 45% of the wrapper width */
    height: 100%;
    text-align: center; /* Center-align text inside each container */
}

/* Heading style */
.container h1 {
    color: #333; /* Dark grey color for the heading */
    font-size: 30px; /* Adjust font size */
    margin-bottom: 5px; /* Space below the heading */
}

/* Paragraph style */
.container p {
    color: #555; /* Medium grey color for the text */
    font-size: 17px; /* Adjust font size */
}


@import url('https://fonts.googleapis.com/css?family=Exo:400,700');

*{
    margin: 0px;
    padding: 0px;
}

body{
    font-family: 'Exo', sans-serif;
}


.context {
    position: absolute;
    top: 0px;
    right: 0px;
    width: 100%;

    
}

.context h1{
    
    text-align: center;
    color: #fff;
    font-size: 50px;
}


.backgroundArea{
    position: fixed;
    top: 0%;
    right: 0px;
    //background-image: url("https://img.freepik.com/free-vector/gradient-background-green-tones_23-2148370439.jpg?semt=ais_hybrid");  
    //background-image: url("https://static.vecteezy.com/system/resources/thumbnails/004/771/729/small/abstract-gradient-background-with-green-color-spotlight-pattern-illustration-free-vector.jpg");
    background-image: url("https://t3.ftcdn.net/jpg/08/85/71/70/360_F_885717036_EWrc1whNI9DYOURgqO0dB0n87JOswiC0.jpg");
    width: 100%;
    height:100%;
    background-size: cover ;
    
    
    }
.area{
    position: absolute;
    top: 0%;
    right: 0px;
    //background-image: url("https://www.publicdomainpictures.net/pictures/490000/nahled/papier-textur-hintergrund-lila-1674871335Ww5.jpg");  
   // background-image: url("https://cdn.pixabay.com/video/2023/07/21/172655-847860558_tiny.jpg");
    background-image: url("https://cdn.pixabay.com/video/2023/07/21/172655-847860558_tiny.jpg");
    width: 100%;
    background-size: 100% auto;
    height:10em;}
    
   

.circles{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
}

.circles li{
    position: absolute;
    display: block;
    list-style: none;
    width: 20px;
    height: 20px;
    background: rgba(255, 255, 255, 0.2);
    animation: animate 25s linear infinite;
    bottom: -150px;
    
}

.circles li:nth-child(1){
    left: 25%;
    width: 80px;
    height: 80px;
    animation-delay: 0s;
}


.circles li:nth-child(2){
    left: 10%;
    width: 20px;
    height: 20px;
    animation-delay: 2s;
    animation-duration: 12s;
}

.circles li:nth-child(3){
    left: 70%;
    width: 20px;
    height: 20px;
    animation-delay: 4s;
}

.circles li:nth-child(4){
    left: 40%;
    width: 60px;
    height: 60px;
    animation-delay: 0s;
    animation-duration: 18s;
}

.circles li:nth-child(5){
    left: 65%;
    width: 20px;
    height: 20px;
    animation-delay: 0s;
}

.circles li:nth-child(6){
    left: 75%;
    width: 110px;
    height: 110px;
    animation-delay: 3s;
}

.circles li:nth-child(7){
    left: 35%;
    width: 150px;
    height: 150px;
    animation-delay: 7s;
}

.circles li:nth-child(8){
    left: 50%;
    width: 25px;
    height: 25px;
    animation-delay: 15s;
    animation-duration: 45s;
}

.circles li:nth-child(9){
    left: 20%;
    width: 15px;
    height: 15px;
    animation-delay: 2s;
    animation-duration: 35s;
}

.circles li:nth-child(10){
    left: 85%;
    width: 150px;
    height: 150px;
    animation-delay: 0s;
    animation-duration: 11s;
}



@keyframes animate {

    0%{
        transform: translateY(0) rotate(0deg);
        opacity: 1;
        border-radius: 0;
    }

    100%{
        transform: translateY(-1000px) rotate(720deg);
        opacity: 0;
        border-radius: 50%;
    }

}
</style>



    

"""
# Access the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai_api_key"]

def extract_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def html_table_to_dataframe(html_table):
    soup = BeautifulSoup(html_table, "html.parser")
    table_data = []

    headers = [header.text for header in soup.find_all("th")]
    rows = soup.find_all("tr")[1:]  # Skip header row
    for row in rows:
        columns = row.find_all("td")
        row_data = [col.text for col in columns]
        table_data.append(row_data)

    df = pd.DataFrame(table_data, columns=headers)
    return df

# Function to evaluate the student's code and generate an HTML table with feedback
def grade_code(file_content, question_pdf_path):
    # Load the question content into the embeddings
    pdf_loader = PyPDFLoader(question_pdf_path)
    pdf_docs = pdf_loader.load()

    # Initialize the OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create the vector store for the PDF documents
    vector_store = FAISS.from_texts([doc.page_content for doc in pdf_docs], embeddings)

    # Set up the retriever and QA chain
    retriever = vector_store.as_retriever()
    fine_tuned_model_id = "ft:gpt-4o-mini-2024-07-18:personal::AS33rYst"
    llm = ChatOpenAI(model_name=fine_tuned_model_id, openai_api_key=openai_api_key)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # Run the QA chain to evaluate the code
    response = qa_chain.run(
        f"Evaluate the student's Python script in the file using the provided grading criteria. "
        f"Generate HTML table code for the student's evaluation with columns for each grading criteria. "
        f"Include feedback in the 'Comments' column such as missing code, incorrect logic, or formatting issues. "
        f"Provide detailed feedback for improvements, and if a student earns full marks in a criterion, add an encouraging phrase like 'Great job!' or 'Excellent work!'. "
        f"Ensure consistency in the structure and clarity in the feedback."
    )
    return response


st.markdown(pageBase, unsafe_allow_html=True)

##############################################################################
#                                   Read logo
##############################################################################
logopath = "logo.png"
file_ = open(logopath, "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

##############################################################################
#                                   Tabs
##############################################################################

st.markdown('<div class="backgroundArea" >  <ul class="circles"> <li></li> <li></li> <li></li> <li></li><li></li> <li></li> <li></li> <li></li> <li></li> <li></li>  </ul></div >', unsafe_allow_html=True) 
tab1, tab2, tab3 = st.tabs(["Who are we?", 
                            "Try MarkIt!", 
                            "FAQ"])

with tab1:
    st.markdown('<div class="area" >  <ul class="circles"> <li></li> <li></li> <li></li> <li></li><li></li> <li></li> <li></li> <li></li> <li></li> <li></li>  </ul></div >', unsafe_allow_html=True)   
    st.markdown(  '<div class="context"> <h1>Who are we?</h1> </div><br/><br/><br/>' , unsafe_allow_html=True)  

##############################################################################
#                                   Containers
##############################################################################
    left, center, right = st.columns(3)
    
    #with center:
    #    st.image(logopath, width=200)
    
    st.markdown('<div class="mainContainer"><p><br />MarkIt is your intelligent teaching assisstant! Mark your codes in seconds and get detailed feedback on howto enhance your coding skills!</p></div><br/>', unsafe_allow_html=True)
         
    
    
    a,b = st.columns(2)
    
    with a:
        st.image(logopath, use_column_width=True)
        st.markdown('''
            <div class="container">
            <h1>Our users?</h1>
                <p>This Whether you are a coding teacher or a student, MarkIt will help you get detailed feedback and performance analyses to assess your codes based on your own criteria and grading table, just as a teaching assistant would. </p>
            </div><br/>''', unsafe_allow_html=True)
        st.write("###")
        st.markdown('''
            <div class="container">
            <h1>Data Privacy? </h1>
                <p>The current version of MarkIt does not save any copies of your submissions or papers. Future versions could be integrated with elearning platforms to include comparisons and capture students' improvement through the assessments. Integration with elearning platforms will also allow MarkIt to understand about the course material and students backgrounds. </p>
            </div><br/>''', unsafe_allow_html=True)

        

    
    with b: 
        st.write("###")
        st.markdown(''' <div class="mainContainer">
             <h1>Background?</h1>
             <p>TMarkIt is trained on real anynomous students assignments, submissions, and marks, to get TA-like feedback and machine-like precision and speed</p>
         </div>''', unsafe_allow_html=True)
        st.write("###")

        st.markdown(''' <div class="mainContainer">
             <h1>Our goal?</h1>
             <p>MarkIt aims to improve students performance by providing the personalized constructive feedback they deserve, and to assist instructors by displaying students performance analyses and help them adapt to students' needs<br></p>
         </div>''', unsafe_allow_html=True)
        #st.write("###")
        impath = "3.png"
        file_ = open(impath, "rb")
        st.image(impath, use_column_width=True)

       # st.markdown(''' <div class="extra"> <br><br><br> </div>''', unsafe_allow_html=True)

    st.markdown('<div class="area" >  <ul class="circles"> <li></li> <li></li> <li></li> <li></li><li></li> <li></li> <li></li> <li></li> <li></li> <li></li>  </ul></div >', unsafe_allow_html=True)   
    
    
with tab2:
    st.markdown(
        '<div class="area"><ul class="circles"><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li><li></li></ul></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="context"><h1>Try MarkIt</h1></div><br/><br/><br/>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="mainContainer"><p><br /><br />MarkIt is your intelligent teaching assisstant! Mark your codes in seconds and get detailed feedback on howto enhance your coding skills!</p></div><br/>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.write("###")
        st.markdown('''
            <div class="mainContainer">
                <h6>Upload Question as PDF</h6>
            </div>
        ''', unsafe_allow_html=True)
        st.write("###")
        st.markdown('''
            <div class="mainContainer">
                <h6>Upload Student Solutions (.py files)</h6>
            </div>
        ''', unsafe_allow_html=True)

    with col2:
        question_pdf = st.file_uploader("Upload PDF with Questions", type="pdf")
        student_solutions = st.file_uploader("Upload Student Solutions (.py)", type="py", accept_multiple_files=True)

    if question_pdf and student_solutions:
        question_pdf_path = question_pdf.name  # Pass file name for PyPDFLoader

        # Save uploaded question PDF locally for PyPDFLoader
        with open(question_pdf_path, "wb") as f:
            f.write(question_pdf.getbuffer())

        st.markdown('''
            <div class="mainContainerGrade">
                <p>Click below to start grading:</p>
            </div>
        ''', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col2:
            st.write("###")
            st.write("###")
            enter = st.button("Enter to Grade")

        if enter:
            for i, student_file in enumerate(student_solutions, start=1):
                file_content = student_file.read().decode("utf-8")
                feedback_html = grade_code(file_content, question_pdf_path)
                
                # Convert HTML table to DataFrame
                feedback_df = html_table_to_dataframe(feedback_html)
                
                # Convert DataFrame to HTML table string
                feedback_html_table = feedback_df.to_html(index=False, escape=False)

                # Display as a full HTML block with a <div class="mainContainer">
                st.markdown(
                    f'<div class="mainContainer"><h3>Feedback for {student_file.name}</h3>{feedback_html_table}</div>',
                    unsafe_allow_html=True
                )



    
with tab3:
    st.markdown('<div class="area" >  <ul class="circles"> <li></li> <li></li> <li></li> <li></li><li></li> <li></li> <li></li> <li></li> <li></li> <li></li>  </ul></div >', unsafe_allow_html=True)   
    st.markdown(  '<div class="context"> <h1>FAQ</h1> </div><br/><br/><br/>' , unsafe_allow_html=True)  
    st.markdown('<div class="mainContainer"><h1>Can I use?</h1><p>MarkIt is your intelligent teaching assisstant! Mark your codes in seconds and get detailed feedback on howto enhance your coding skills!</p></div><br/>', unsafe_allow_html=True)
    st.markdown('<div class="mainContainer"><h1>Can I link MarkIt to Moodle?</h1><p>MarkIt is your intelligent teaching assisstant! Mark your codes in seconds and get detailed feedback on howto enhance your coding skills!</p></div><br/>', unsafe_allow_html=True)
    st.markdown('<div class="mainContainer"><h1>How will I be </h1><p>MarkIt is your intelligent teaching assisstant! Mark your codes in seconds and get detailed feedback on howto enhance your coding skills!</p></div><br/>', unsafe_allow_html=True)


#image1 ="https://img.freepik.com/free-vector/dark-polygonal-background_79603-282.jpg"
#st.image(image1, use_column_width='auto')
