import bisect

import cv2
import numpy as np
import sys
import math
from scipy import stats
import sakuya


class Staff:
	treblePitches = (
	'C6', 'B5', 'A5', 'G5', 'F5', 'E5', 'D5', 'C5', 'B4', 'A4', 'G4', 'F4', 'E4', 'D4', 'C4', 'B3', 'A3')
	bassPitches = ('E4', 'D4', 'C4', 'B3', 'A3', 'G3', 'F3', 'E3', 'D3', 'C3', 'B2', 'A2', 'G2', 'F2', 'E2', 'D2', 'C1')

	def __init__(self, lineSpacing=0, lineThickness=0, tops=[], timeSignatures=[], keySignatures=[]):
		self.lineSpacing = lineSpacing
		self.lineThickness = lineThickness
		self.tops = tops
		self.keySignatures = keySignatures
		self.timeSignatures = timeSignatures
		self.trebles = []
		self.basses = []
		self.lines = {}
		self.initLines()

	# Assumes that the first value in top corresponds to a treble staff and the rest alternate between bass and treble

	def initLines(self):
		treble = True
		for top in self.tops:
			if (treble):
				self.trebles.append(top)
				treble = False
			else:
				self.basses.append(top)
				treble = True

		treble = True
		pitches = self.treblePitches
		lineDifference = self.lineSpacing + self.lineThickness
		lineDifferenceIsOdd = True if (lineDifference % 2 == 1) else False
		for top in self.tops:
			currentY = top - 2 * (lineDifference) + self.lineThickness / 2
			plusOne = False
			for pitch in pitches:
				self.lines[currentY] = pitch
				currentY = currentY + lineDifference / 2
				# This is used in the case that lineDifference is odd and we must compensate by adding 1 every other line to ensure corrent line spacing
				if (lineDifferenceIsOdd):
					currentY = currentY + (1 if plusOne else 0)
					plusOne = not (plusOne)
			treble = (not (treble))
			if (treble):
				pitches = self.treblePitches
			else:
				pitches = self.bassPitches
		print(self.trebles)
		print(self.basses)
		print(self.lines)

	def getPitch(self, y):
		yValues = self.lines.keys()
		yValues.sort()
		key = bisect.bisect_left(yValues, y)
		if (not (key == 0)):
			distanceAbove = y - yValues[key - 1]
			distanceBelow = yValues[key] - y
			if (distanceAbove < distanceBelow):
				key = key - 1
		if (key == len(yValues)):
			key = key - 1
		return self.lines[yValues[key]]

# Returns the mode vertical run length of the given colour in the input image
def verticalRunLengthMode(img,colour):
	runLengths = []
	width = len(img[0])
	height = len(img)
	for x in range(int(width*1/4),int(width*3/4)):
		inColour = False
		currentRun = 0
		for y in range(0,height):
			if (img[y,x] == colour):
				if (inColour):
					currentRun = currentRun + 1
				else:
					currentRun = 1
					inColour = True
			else:
				if (inColour):
					runLengths.append(currentRun)
					inColour = False

	return int(stats.mode(runLengths)[0][0])

# Returns True iff rho corresponds to the top of a set of 5 staff lines
def isTopStaffLine(rho,rhoValues,gap,threshold):
	for i in range(1,5):
		member = False
		for j in range(0,threshold+1):
			if ((rho + i*gap + j) in rhoValues or (rho + i*gap - j) in rhoValues):
				member = True
		if (not(member)):
			return False
	return True

# Input: a thresholded sheet music image
# Output: an omr_classes.Staff object corresponding to the input
def getStaffData(imgInput):
	width = len(imgInput[0])
	height = len(imgInput)


	imgBinary = imgInput

	# Output binary image

	imageName = f'staff'
	cv2.imwrite('binary_output_' + imageName + '.png',imgBinary)

	# Apply Hough line transform
	lines = cv2.HoughLines(imgBinary,1,np.pi/180,int(min(width, height)/2))


	# Show lines on copy of input image
	imgHoughLines = cv2.imread(f'w-21_p008.png')

	for i in range(0, len(lines)):
		rho = lines[i][0][0]
		theta = lines[i][0][1]
		a = math.cos(theta)
		b = math.sin(theta)
		x0 = a * rho
		y0 = b * rho
		pt1 = (int(x0 + width * (-b)), int(y0 + height * (a)))
		pt2 = (int(x0 - width * (-b)), int(y0 - height * (a)))
		cv2.line(imgHoughLines, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

	# Output image with Hough lines
	imageName = f'hugh'
	cv2.imwrite('hough_output_' + imageName + '.png',imgHoughLines)

	# Find most common black pixel run length (white pixel run length in binary image due to inversion)
	# This should correspond to staff line thickness
	staffLineThickness = verticalRunLengthMode(imgBinary,255)
	print("staffLineThickness: " + str(staffLineThickness))

	# Find staff line spacing

	# Find average difference in rho

	sortedHoughLines = sorted(lines,key = lambda x : x[0][0])


	staffLineSpacing = verticalRunLengthMode(imgBinary,0) + staffLineThickness
	print("staffLineSpacing: " + str(staffLineSpacing))
	# Now we keep only lines with theta between 1.56 and 1.58. We also only store rho values from now on

	sortedRhoValues = []
	for i in sortedHoughLines:
		rho = i[0][0]
		theta = i[0][1]
		if (1.56 < theta and theta < 1.58):
			sortedRhoValues.append(int(rho))

	print("sortedRhoValues: " + str(sortedRhoValues))

	# Find rho value of top line of each stave
	staffTops = []

	for rho in sortedRhoValues:
		if (isTopStaffLine(rho,sortedRhoValues,staffLineSpacing,5)):
			staffTops.append(rho)

	print("staffTops: " + str(staffTops))

	# Show staff lines on copy of input image
	imgStaffLines = imgInput.copy()
	imgStaffLines = cv2.cvtColor(imgStaffLines,cv2.COLOR_GRAY2RGB)
	for rho in staffTops:
		for i in range(0,5):
			y = rho + i*staffLineSpacing
			cv2.line(imgStaffLines,(0,y),(width-1,y),(0,0,255),1)

	# Output image with staff lines
	imageName = 'staff'
	cv2.imwrite('staff_output_' + imageName + '.png',imgStaffLines)

	currentStaff = Staff(staffLineSpacing,staffLineThickness,staffTops)
	return currentStaff

x = getStaffData(sakuya.binarize(f'w-21_p008.png'))
