from email import message
import cv2 as cv
import numpy as np
import tensorflow as tf


def initializePredictionModel():
  json_file = open('model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = tf.keras.models.model_from_json(loaded_model_json)
  model.load_weights("model_trained.h5")

  return model

heightImage = 450
widthImage = 450
imageDimensions = (32, 32, 3)

def biggestContour(contours):
  biggest = contours[0]
  max_area = 0
  for c in contours:
    area = cv.contourArea(c)
    if area > max_area:
      biggest = c
      max_area = area
  return biggest, max_area

def reorder(contour):
  max_bottom_right = 0
  min_top_left = 1e100
  min_bottom_left = 1e400
  max_top_right = 0

  bottom_right = []
  bottom_left = []
  top_left = []
  top_right = []
  for point in contour:
      if(point[0][1] + point[0][0] > max_bottom_right):
          max_bottom_right = point[0][1] + point[0][0]
          bottom_right = [point[0][0], point[0][1]]

      if(point[0][1] + point[0][0] < min_top_left):
          min_top_left = point[0][1] + point[0][0]
          top_left = [point[0][0], point[0][1]]

      if(point[0][0] - point[0][1] > max_top_right):
          max_top_right = point[0][0] - point[0][1]
          top_right = [point[0][0], point[0][1]]

      if(point[0][0] - point[0][1] < min_bottom_left):
          min_bottom_left = point[0][0] - point[0][1]
          bottom_left = [point[0][0], point[0][1]]

  coordinates = [[top_left], [top_right], [bottom_left], [bottom_right]]
  coordinates = np.array(coordinates, dtype="float32")
  
  return coordinates

def splitBoxes(img):
  rows = np.vsplit(img, 9)
  boxes = []
  for r in rows:
    cols = np.hsplit(r, 9)
    for box in cols:
      boxes.append(box)
  return boxes

def getPrediction(boxes, model):
  result = []
  for image in boxes:
    img = np.asarray(image)
    img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
    img = cv.resize(img, (imageDimensions[0], imageDimensions[1]))
    img = img/255
    img = img.reshape(1, imageDimensions[0], imageDimensions[1], 1)

    predictions = model.predict(img)

    classIndex = np.argmax(predictions, axis = -1)
    probabilityValue = np.amax(predictions)
    if probabilityValue > 0.35:
      result.append(classIndex[0])
      ans = classIndex[0]
    else:
      result.append(0)
      ans = 0
  return result 

def displayNumbers(numbers, color = (0,255,0)):
  imgBlank = 255*np.ones((heightImage,widthImage,3),np.uint8)
  secW = int(imgBlank.shape[1]/9)
  secH = int(imgBlank.shape[0]/9)
  for x in range(0,9):
    for y in range(0,9):
      if(numbers[(y*9) + x] != 0):
        cv.putText(imgBlank, str(numbers[(y*9) + x]), 
                   (x*secW + int(secW/2)-10, int((y + 0.8)*secH)), 
                   cv.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1)
  for x in range(0,9):
    thickness = 2
    if(x%3 == 0):
      thickness = 3
    cv.line(imgBlank, (x*secW,0), (x*secW, imgBlank.shape[0]), (0,0,0), thickness)
    cv.line(imgBlank, (0,x*secH), (imgBlank.shape[1], x*secH), (0,0,0), thickness)

  cv.line(imgBlank, (imgBlank.shape[1],0), (imgBlank.shape[1], imgBlank.shape[0]), (0,0,0), 3)
  cv.line(imgBlank, (0,imgBlank.shape[0]), (imgBlank.shape[1], imgBlank.shape[0]), (0,0,0), 3)

  return imgBlank

def preProcess(img):
  imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  # imgHist = cv.equalizeHist(imgGray)
  imgBlur = cv.GaussianBlur(imgGray,(5,5),1)
  imgThreshold = cv.adaptiveThreshold(imgBlur,255,1,1,11,2)
  return imgThreshold

def startProcessing(imgThreshold, img):
  imgContours = img.copy()
  imgBlank = np.zeros((heightImage,widthImage,3),np.uint8)
  contours, hierarchy = cv.findContours(imgThreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  cv.drawContours(imgContours, contours, -1 , (0,255,0),3)
  biggest, maxArea = biggestContour(contours)

  if len(biggest) != 0:
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[widthImage,0],[0,heightImage],[widthImage,heightImage]])
    matrix = cv.getPerspectiveTransform(pts1,pts2)
    imgWarpColored = cv.warpPerspective(img,matrix, (widthImage,heightImage))
    imgDetectedDigits = imgBlank.copy()
    imgWarpGray = cv.cvtColor(imgWarpColored, cv.COLOR_BGR2GRAY)
    return imgWarpGray
  
  return imgBlank

def getSudokuNumbers(imgWarpGray, model):
  imgBlank = 255*np.ones((heightImage,widthImage,3),np.uint8)
  imgSolvedDigits = imgBlank.copy()
  boxes = splitBoxes(imgWarpGray)
  numbers = getPrediction(boxes, model)
  numbers = np.asarray(numbers)
  return numbers

def check(row, col, val, grid):
  # Checking in the row
  for y in range(0,9):
    if(grid[row][y] == val):
      return False
  # Checking in the col
  for x in range(0,9):
    if(grid[x][col] == val):
      return False
  topR = (row//3)*3
  lowR = (row//3 + 1)*3
  lftC = (col//3)*3
  rgtC = (col//3 + 1)*3

  for x in range(topR, lowR):
    for y in range(lftC, rgtC):
      if(grid[x][y] == val):
        return False

  return True

def rec(row, col, grid):
  if(row == 9):
    return True

  if(grid[row][col] != 0):
    if(col == 8):
      return rec(row+1,0, grid)
    else:
      return rec(row,col+1, grid)
  else:
    for val in range(1,10):
      if(check(row,col, val, grid)):
        grid[row][col] = val
        if(col == 8):
          if(rec(row+1,0, grid)):
            return True
        else:
          if(rec(row, col+1, grid)):
            return True
        grid[row][col] = 0

  return False

def displaySolvedSudoku(img, sudoku, solvedSudoku, isSolved, color1, color2):
  secW = int(img.shape[1]/9)
  secH = int(img.shape[0]/9)
  for x in range(0,9):
    for y in range(0,9):
      if(sudoku[x][y] != 0):
        cv.putText(img, str(sudoku[x][y]), 
                   (int((y + 0.5)*secH)-10, x*secW + int(secW/2) +10 ), 
                   cv.FONT_HERSHEY_COMPLEX_SMALL, 1, color1, 1)
      elif(isSolved):
        cv.putText(img, str(solvedSudoku[x][y]), 
                   (int((y + 0.5)*secH - 10), x*secW + int(secW/2) + 10), 
                   cv.FONT_HERSHEY_COMPLEX_SMALL, 1, color2, 2)
  for x in range(0,9):
    thickness = 2
    if(x%3 == 0):
      thickness = 3
    cv.line(img, (x*secW,0), (x*secW, img.shape[0]), (0,0,0), thickness)
    cv.line(img, (0,x*secH), (img.shape[1], x*secH), (0,0,0), thickness)

  cv.line(img, (img.shape[1],0), (img.shape[1], img.shape[0]), (0,0,0), 3)
  cv.line(img, (0,img.shape[0]), (img.shape[1], img.shape[0]), (0,0,0), 3)

  return img

def solve(numbers):
  imgBlank = 255*np.ones((heightImage,widthImage,3),np.uint8)
  sudoku = numbers.copy()
  sudoku.shape = (9,9)
  grid = sudoku.copy()
  isSolved = rec(0,0,grid)
  solvedSudokuImg = imgBlank.copy()
  message = "Incorrect sudoku"
  displaySolvedSudoku(solvedSudokuImg, sudoku, grid, isSolved, (0, 0, 0), (255,0,0))
  if(isSolved):
    message = "Solved Sudoku"

  return solvedSudokuImg, message
      
