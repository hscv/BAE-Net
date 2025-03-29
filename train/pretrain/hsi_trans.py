import numpy as np
def X2Cube(img):
        img = np.asarray(img)
        B = [4, 4]
        skip = [4, 4]
        # Parameters
        M, N = img.shape
        col_extent = N - B[1] + 1
        row_extent = M - B[0] + 1

        # Get Starting block indices
        start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])

        # Generate Depth indeces
        didx = M * N * np.arange(1)
        start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))

        # Get offsetted indices across the height and width of input array
        offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)

        # Get all actual indices & index into input array for final output
        out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
        out = np.transpose(out)
        # print ('out.shape = ',out.shape)
        # print ('M = ',M,' , N = ',N)
        img = out.reshape(M//4, N//4, 16)
        # print ('img.shape = ',img.shape)
        img = img.transpose(1,0,2)
        # print ('---222---img.shape = ',img.shape)
        img = img / img.max() * 255 #  归一化
        img.astype('uint8')
        return img
