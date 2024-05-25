import random
import time

# Function to generate random values between -1 and 1
def generateRandomValues(shape):
    matrix = []
    for _ in range(shape[0]):
        dimension1 = []
        for _ in range(shape[1]):
            dimension2 = []
            for _ in range(shape[2]):
                dimension3 = []
                for _ in range(shape[3]):
                    value = random.random()
                    dimension3.append(value)
                dimension2.append(dimension3)
            dimension1.append(dimension2)
        matrix.append(dimension1)
    return matrix

# Calculate output dimensions
def outputDimension(H, R, W, S, stride):
    Hout = (H - R) // stride + 1
    Wout = (W - S) // stride + 1
    return Hout, Wout

# Padding before pooling operation
def Padding(N, M, Hout, Wout, outputFMAP):
    for x in range(N):
        for y in range(M):
            HoutNew = Hout
            if len(outputFMAP[x][y]) % 2 == 1:
                outputFMAP[x][y].append([0 for _ in range(Wout)])
                HoutNew += 1
            if len(outputFMAP[x][y][0]) % 2 == 1:
                for z in range(HoutNew):
                    outputFMAP[x][y][z].append(0)
    return outputFMAP

# average pooling (assuming a simple 2x2 pooling window with stride 2)
def averagePooling(featureMap, poolSize=2, stride=2):
    N, M, H, W = len(featureMap), len(featureMap[0]), len(featureMap[0][0]), len(featureMap[0][0][0])
    Hout = (H - poolSize) // stride + 1
    Wout = (W - poolSize) // stride + 1
    outputFMapPooled = [[[
        [0 for _ in range(Wout)]
        for _ in range(Hout)] 
        for _ in range(M)] 
        for _ in range(N)]
    
    for n in range(N):
        for m in range(M):
            for h in range(Hout):
                for w in range(Wout):
                    hStart = h * stride
                    wStart = w * stride
                    hEnd = hStart + poolSize
                    wEnd = wStart + poolSize
                    pool_sum = 0
                    for i in range(hStart, hEnd):
                        for j in range(wStart, wEnd):
                            pool_sum += featureMap[n][m][i][j]
                    outputFMapPooled[n][m][h][w] = pool_sum / (poolSize * poolSize)
    return outputFMapPooled

# Function to compare two nested lists (feature maps)
def compareFeatureMaps(fm1, fm2):
    for i in range(len(fm1)):
        for j in range(len(fm1[0])):
            for k in range(len(fm1[0][0])):
                for l in range(len(fm1[0][0][0])):
                    if abs(fm1[i][j][k][l] - fm2[i][j][k][l]) > 1e-6:  # Allow small numerical differences
                        return False
    return True

# 7-loop naive implementation of convolution
def SevenLoopNaiveMethod(N, C, H, W, R, S, M, stride, inputFMap, weights):
    Hout, Wout = outputDimension(H, R, W, S, stride)
    # Initialize output feature maps
    outputFMAP = [[[[0 for _ in range(Wout)] for _ in range(Hout)] for _ in range(M)] for _ in range(N)]
    # Timer Start
    start_time = time.time()
    for n in range(N):  # batch size
        for m in range(M):  # output channels
            for x in range(Hout):  # output height
                for y in range(Wout):  # output width

                    for c in range(C):  # input channels
                        for r in range(R):  # filter height
                            for s in range(S):  # filter width
                                outputFMAP[n][m][x][y] += inputFMap[n][c][x * stride + r][y * stride + s] * weights[m][c][r][s]
    end_time = time.time()
    naiveConvTime = end_time - start_time
    outputFMapPadded = Padding(N, M, Hout, Wout, outputFMAP)
    # Apply average pooling to the output feature maps
    outputFMapPooled = averagePooling(outputFMapPadded)
    return naiveConvTime, outputFMapPooled

# Flattening Filter Map
def flattenWeights(weights, M, C, R, S):
    weightMatrix = []
    for m in range(M):
        row = []
        for c in range(C):
            for r in range(R):
                for s in range(S):
                    row.append(weights[m][c][r][s])
        weightMatrix.append(row)
    return weightMatrix

# Flattening input Feature Map
def flattenInput(inputFMap, N, C, R, S, Hout, Wout, stride):
   # Initialize flattened input list
    flattenedInput = [[] for _ in range(R*S*C)]
    for n in range(N):  # batch size
        for c in range(C):  # input channels
            for x in range(Hout):  # output height
                for y in range(Wout):  # output width 
                    for r in range(R):  # filter height
                        for s in range(S):  # filter width
                            flattenedInput[c*(R*S)+r*R+s].append(inputFMap[n][c][x * stride + r][y * stride + s])

    # flattened input R*S x Hout*Wout*C*N
    return flattenedInput, Hout, Wout

# Multiply flattened matrix
def MultiplyMatrix(M, inputMatrix, weightMatrix):
    outputMatrix = [[0 for _ in range(len(inputMatrix[0]))] for _ in range(M)]
    for i in range(M):
        for j in range(len(inputMatrix[0])):
            for k in range(len(inputMatrix)):
                outputMatrix[i][j] += weightMatrix[i][k] * inputMatrix[k][j]
    return outputMatrix

# Reshape the output matrix to the original output shape
def outputFMapFromMatrix(N, M,Hout, Wout,outputMatrix):
    outputFMap = [[[[0 for _ in range(Wout)] for _ in range(Hout)] for _ in range(M)] for _ in range(N)]
    for n in range(N):
        for m in range(M):
            for h in range(Hout):
                for w in range(Wout):
                    outputFMap[n][m][h][w] = outputMatrix[m][n*(Hout*Wout)+h*Hout+w]
    return outputFMap

# Flattening method
def FlatteningMethod(N, C, H, W, R, S, M, stride, inputFMap, weights):
    Hout, Wout = outputDimension(H, R, W, S, stride)
    # Perform convolution by flattening
    start_time = time.time()
    inputMatrix, Hout, Wout = flattenInput(inputFMap, N, C, R, S, Hout, Wout, stride)
    weightMatrix = flattenWeights(weights, M, C, R, S)
    outputMatrix = MultiplyMatrix(M, inputMatrix, weightMatrix)
    outputFMap = outputFMapFromMatrix(N, M, Hout, Wout, outputMatrix)
    end_time = time.time()
    flattenedConvTime = end_time - start_time
    outputFMapPadded = Padding(N, M, Hout, Wout, outputFMap)
    # Apply average pooling to the output feature maps
    outputFeatureMapFlattenedPooled = averagePooling(outputFMapPadded)
    return flattenedConvTime, outputFeatureMapFlattenedPooled

# Parameters
N, C, H, W = 10, 3, 28, 28
R, S, M = 3, 3, 64
stride = 2
# Generate random input feature maps and weights
inputFMap = generateRandomValues((N, C, H, W))
weights = generateRandomValues((M, C, R, S))

print("--------Convolution by Flattening-------")
flattenedConvTime, outputFeatureMapFlattenedPooled = FlatteningMethod(N, C, H, W, R, S, M, stride, inputFMap, weights)
# Print time taken for convolution by flattening
print("Time taken for convolution by flattening:", flattenedConvTime)
# Check the shape of the output feature maps
print("Output feature maps shape after pooling: (N, M, H_out, W_out) =", len(outputFeatureMapFlattenedPooled), len(outputFeatureMapFlattenedPooled[0]), len(outputFeatureMapFlattenedPooled[0][0]), len(outputFeatureMapFlattenedPooled[0][0][0]))
print("----------------------------------------")    
# Compare the output feature maps
outputs_equal = compareFeatureMaps(outputFMapPooled, outputFeatureMapFlattenedPooled)
print("Both methods produce the same output?:", outputs_equal)