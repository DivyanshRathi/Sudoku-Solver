# from turtle import onclick
import streamlit as st
import cv2 as cv
import numpy as np
import sudokudetector as sd

st.set_page_config(initial_sidebar_state="expanded", layout="wide")
numbers = np.asarray([0]*81)
container1 = st.container()
container2 = st.container()
detectedSudoku = st.empty()
solvedSudoku = st.empty()

original_image = None
preProcessed_image = None
options = range(0,10)
imageDim = sd.imageDimensions
showUpdateForm = False

if "row" in st.session_state:
    showUpdateForm = True
    numbers = st.session_state.numbers
    row = st.session_state.row
    col = st.session_state.col
    value = st.session_state.value
    if(row > 0 and col > 0):
        numbers[9*row-9 + col-1] = value
        st.session_state.numbers = numbers
    


def showSudokuSolved():
    showUpdateForm = False
    detectedSudoku.empty()
    with solvedSudoku.container():
        st.subheader("Solved Sudoku")
        st.image(sd.solve(numbers)[0])



with st.sidebar:
    st.title("Sudoku-Solver")
    uploaded_file = st.file_uploader("Choose image", type = ["png", "jpg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv.imdecode(file_bytes, 1)


if (original_image is not None):
    model = sd.initializePredictionModel()
    original_image = cv.resize(original_image,(sd.widthImage,sd.heightImage))
    preProcessed_image = sd.preProcess(original_image.copy())
    imgWarpGray = sd.startProcessing(preProcessed_image, original_image)

    with container1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.header("Original image")
            st.image(original_image, channels="BGR")

        with col2:
            st.header("Isolated Sudoku")
            st.image(imgWarpGray)
    if "numbers" not in st.session_state:
        showUpdateForm = True
        numbers = np.asarray([0]*81)
        numbers = sd.getSudokuNumbers(imgWarpGray, model) 
        st.session_state.numbers = numbers
        
    imgDetectedDigits = sd.displayNumbers(numbers, color = (0,0,0))

if (original_image is not None):
    with detectedSudoku.container():
        st.header("Detected Sudoku Numbers")
        st.image(imgDetectedDigits)

solvedSudoku.empty()

if showUpdateForm:
    with st.sidebar:
        st.subheader("Update incorrect values")
        form = st.form("my_form", True)
        with form:
            col1, col2, col3 = st.columns(3)

            with col1:
                row = st.selectbox("Select row", options, key="row")
            
            with col2:
                col = st.selectbox("Select column", options, key="col")
            
            with col3:
                value = st.selectbox("Select value", options, 0, key = "value")
        
        submitted = form.form_submit_button("Submit")
        
        solve = st.button("Solve")
        if(solve):
            showSudokuSolved()
