from sklearn.preprocessing import LabelEncoder

#this function is use to convert text data into numeric form
def encoder(dataset,indices):
    encoder = LabelEncoder()
    X_trans= dataset.iloc[:, indices].values
    X_trans = encoder.fit_transform(X_trans)
    return X_trans

#This function is used to save file as xls
def Write_XL(List,row,col,category): 
    from xlwt import Workbook

    wb = Workbook() 
    sheet1 = wb.add_sheet('Sheet 1')
    
    #This loop is for writing header
    for i in range(col):
        sheet1.write(0,i,category[i])

    for i in range(row):
        for j in range(col):
            sheet1.write(i+1,j,List[i][j])

    wb.save('centroids.xls')