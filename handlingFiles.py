import csv


def readCSV(dataFilePath):
    data = []
    file = open(dataFilePath, 'r')
    csvReader = csv.reader(file)
    for row in csvReader:
        data.append(row)
    data = [[float(element) for element in index] for index in data]
    file.close()
    return data


def writeCSV(dataFilePath, data):
    file = open(dataFilePath, 'w')
    csvWriter = csv.writer(file, lineterminator='\n')
    csvWriter.writerows(data)
    file.close()




