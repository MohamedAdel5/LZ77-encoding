import cv2
import numpy as np
import time
import math


def readImage(inputImageFileName):
    imgMatrix = cv2.imread(inputImageFileName, cv2.IMREAD_GRAYSCALE)
    return imgMatrix


def flattenImage(img):
    return img.flatten()

def getSuitableNumpyDtype(bitNum):
    dataType = None
    if bitNum <= 8:
        dataType = np.uint8
    elif bitNum <= 16:
        dataType = np.uint16
    elif bitNum <= 32:
        dataType = np.uint32
    elif bitNum <= 64:
        dataType = np.uint64
    return dataType

def toNumpyFiles(encodedList, encodedFileName):
    maxOfOffsets = encodedList[0][0]
    maxOfLengths = encodedList[0][1]
    maxOfCodes = encodedList[0][2]
    for triple in encodedList:
        if (triple[0] > maxOfOffsets):
            maxOfOffsets = triple[0]
        if (triple[1] > maxOfLengths):
            maxOfLengths = triple[1]
        if (triple[2] > maxOfCodes):
            maxOfCodes = triple[2]
    maxOfOffsetsAndLengths = max([maxOfOffsets, maxOfLengths])

    bitNum1 = math.ceil(math.log2(maxOfOffsetsAndLengths))
    bitNum1 = 8 if bitNum1 < 8 else bitNum1
    bitNum2 = math.ceil(math.log2(maxOfCodes))
    bitNum2 = 8 if bitNum2 < 8 else bitNum2
    if bitNum1 == bitNum2:
        dtype = getSuitableNumpyDtype(bitNum1)
        encodedNpArray = np.array(encodedList, dtype=dtype).flatten()
        np.save(encodedFileName, encodedNpArray)
        NumberOfEncodedFiles = 1
    else:
        dtype1 = getSuitableNumpyDtype(bitNum1)
        dtype2 = getSuitableNumpyDtype(bitNum2)
        encodedOffsetsAndLengths = np.array(encodedList, dtype=dtype1)[:, :2].flatten()
        encodedCodes = np.array(encodedList, dtype=dtype2)[:, 2:].flatten()
        encodedFileName1 = encodedFileName[:-4] + '1' + encodedFileName[-4:]
        encodedFileName2 = encodedFileName[:-4] + '2' + encodedFileName[-4:]
        np.save(encodedFileName1, encodedOffsetsAndLengths)
        np.save(encodedFileName2, encodedCodes)
        NumberOfEncodedFiles = 2

    return NumberOfEncodedFiles

def loadNumpyEncodedFiles(encodedFileName, NumberOfEncodedFiles):
    if NumberOfEncodedFiles == 1:
        encodedArr = np.load(encodedFileName, allow_pickle=True).reshape(-1, 3)
    else:
        encodedFileName1 = encodedFileName[:-4] + '1' + encodedFileName[-4:]
        encodedFileName2 = encodedFileName[:-4] + '2' + encodedFileName[-4:]

        offsetsAndLengthsArr = np.load(encodedFileName1, allow_pickle=True).reshape(-1, 2)
        codesArr = np.load(encodedFileName2, allow_pickle=True).reshape(-1, 1)
        encodedArr = np.concatenate((offsetsAndLengthsArr, codesArr), axis=1)
    return encodedArr
def LZ77_encode(inputArr, slidingWindowSize, lookAheadBufferSize):
    searchBufferSize = slidingWindowSize - lookAheadBufferSize
    inputArrSize = len(inputArr)
    if inputArrSize <= 1:
        return
    encodedTripletsArr = []
    maxOffset = 0
    searchPtr = 0
    lookAheadPtr = 1
    lookAheadPtrBase = 1
    searchPtrBase = 0
    encodedTripletsArr.append([0, 0, inputArr[0]])  # First element

    while lookAheadPtrBase < inputArrSize:
        lookAheadPtr = lookAheadPtrBase
        searchPtrBase = lookAheadPtrBase - 1
        searchPtr = searchPtrBase

        sequenceTriplet = [None] * 3
        offset = 1

        while searchPtr >= 0 and (offset <= searchBufferSize):
            lookAheadPtr = lookAheadPtrBase

            matchLength = 0
            while (lookAheadPtr < inputArrSize
                   and (searchPtr <= searchPtrBase + lookAheadBufferSize)
                   and (inputArr[lookAheadPtr] == inputArr[searchPtr])):
                matchLength += 1
                searchPtr += 1
                lookAheadPtr += 1
            if sequenceTriplet[1] is None or matchLength > sequenceTriplet[1]:
                sequenceTriplet[0] = offset if matchLength != 0 else 0
                sequenceTriplet[1] = matchLength
                sequenceTriplet[2] = inputArr[lookAheadPtr] if lookAheadPtr < inputArrSize else 0

            searchPtr = searchPtrBase - offset
            offset += 1

        encodedTripletsArr.append(sequenceTriplet)

        lookAheadPtrBase += sequenceTriplet[1] + 1

    return encodedTripletsArr


def LZ77_decode(encodedTripletsArr):

    decodedArr = []

    for triplet in encodedTripletsArr:
        offset = triplet[0]
        matchLength = triplet[1]
        lastSymbolCode = triplet[2]

        decodedArrSize = len(decodedArr)

        for i in range(matchLength):
            matchIndex = decodedArrSize - offset + i % matchLength
            decodedArr.append(decodedArr[matchIndex])

        decodedArr.append(lastSymbolCode)
    return decodedArr


def main():
    slidingWindowSize = 1024
    lookAheadBufferSize = 512
    inputImageFileName = 'inputImg2.png'
    encodedFileName = 'encodedImg.npy'
    decodedFileName = 'decodedImg.png'
    NumberOfEncodedFiles = None
    Y = None
    X = None

    # Step 1: Read the image
    imageMatrix = readImage(inputImageFileName)
    Y = len(imageMatrix)
    X = len(imageMatrix[0])

    # Step 2: Flatten the image
    flattenedImage = flattenImage(imageMatrix)
	
    # Step 3: Encode using LZ77
    print('Started Encoding Please Wait...')
    startTime = time.time()
    encodedList = LZ77_encode(flattenedImage, slidingWindowSize, lookAheadBufferSize)
    print("--- Finished Encoding in %s seconds ---" % (time.time() - startTime))
    print('Encoded List' + str(encodedList))
    
    # # Step 4: Convert list to numpy array and output it to a ".npy" file
    NumberOfEncodedFiles = toNumpyFiles(encodedList, encodedFileName)
    
    ###########################################################
    #                    FINISHED ENCODING
    ###########################################################
    
    print('...')
    print('...')
    print('...')
    
    # Step 6: Decode the encoded file using LZ77.
    encodedArr = loadNumpyEncodedFiles(encodedFileName, NumberOfEncodedFiles)
    
    startTime = time.time()
    decodedArr = np.array(LZ77_decode(encodedArr))
    if len(decodedArr) > Y*X:
        decodedArr = decodedArr[:-1]
    
    print('Started Decoding Please Wait...')
    print("--- Finished Decoding in %s seconds ---" % (time.time() - startTime))
    
    # Step 7: Convert numpy array to image and output it to an image file.
    decodedImageMatrix = decodedArr.reshape(Y, X)
    cv2.imwrite(decodedFileName, decodedImageMatrix.astype(np.uint8))
    
    ###########################################################
    #                 FINISHED DECODING
    ###########################################################

main()
