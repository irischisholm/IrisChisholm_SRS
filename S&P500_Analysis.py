import csv
import statistics

#THIS PROGRAM CALCULATES THE MEAN, MEDIAN, MODE FOR 2 OUT OF 500 S&P 500 STOCKS:
    #STOCK 1: AAL (American Airlines Group Inc) STOCK OVER 5 YEARS (2013-2018)
    #STOCK 2: AAPL (Apple Inc) STOCK OVER 5 YEARS (2013-2018)

AAL = "AAL" #sets up later if statement, which only calculates for AAL out of 500 stocks
AAPL = "AAPL"

with open('/Users/irisyura/Downloads/all_stocks_5yr.csv')as file:
    dataset = csv.reader(file)
    header = next(dataset) #skips the first row bcs it's just column names

    open_index = header.index('open') #finds the column number of "open"
    high_index = header.index('high')
    low_index = header.index('low')
    close_index = header.index('close')
    volume_index = header.index('volume')
    name_index = header.index('Name')

    open_values_AAL = [] #stores the AAL numerical values
    high_values_AAL = []
    low_values_AAL = []
    close_values_AAL = []
    volume_values_AAL = []
    
    open_values_AAPL = [] #stores the AAPL numerical values
    high_values_AAPL = []
    low_values_AAPL = []
    close_values_AAPL = []
    volume_values_AAPL = []

    for lines in dataset:
        value = lines[open_index]
        if value.strip() and lines[name_index] == AAL: #only proceeds if the line is not empty (AKA has a valid numerical value) 
            open_values_AAL.append(float(lines[open_index])) #opens all values from each row of the 'open' column (which was calculated by open_index) and converts that string into a number, then adds it to the open_values list
            high_values_AAL.append(float(lines[high_index]))
            low_values_AAL.append(float(lines[low_index]))
            close_values_AAL.append(float(lines[close_index]))
            volume_values_AAL.append(float(lines[volume_index]))
        if value.strip() and lines[name_index] == AAPL:
            open_values_AAPL.append(float(lines[open_index])) 
            high_values_AAPL.append(float(lines[high_index]))
            low_values_AAPL.append(float(lines[low_index]))
            close_values_AAPL.append(float(lines[close_index]))
            volume_values_AAPL.append(float(lines[volume_index]))
        

open_mean1 = statistics.mean(open_values_AAL) #calculates AAL mean
high_mean1 = statistics.mean(high_values_AAL)
low_mean1 = statistics.mean(low_values_AAL)
close_mean1 = statistics.mean(close_values_AAL)
volume_mean1 = statistics.mean(volume_values_AAL)

open_median1 = statistics.median(open_values_AAL) #calculates AAL median
high_median1 = statistics.median(high_values_AAL)
low_median1 = statistics.median(low_values_AAL)
close_median1 = statistics.median(close_values_AAL)
volume_median1 = statistics.median(volume_values_AAL)

open_mode1 = statistics.mode(open_values_AAL) #calculates AAL mode
high_mode1 = statistics.mode(high_values_AAL)
low_mode1 = statistics.mode(low_values_AAL)
close_mode1 = statistics.mode(close_values_AAL)
volume_mode1 = statistics.mode(volume_values_AAL)



open_mean2 = statistics.mean(open_values_AAPL) #calculates AAPL mean
high_mean2 = statistics.mean(high_values_AAPL)
low_mean2 = statistics.mean(low_values_AAPL)
close_mean2 = statistics.mean(close_values_AAPL)
volume_mean2 = statistics.mean(volume_values_AAPL)

open_median2 = statistics.median(open_values_AAPL) #calculates AAPL median
high_median2 = statistics.median(high_values_AAPL)
low_median2 = statistics.median(low_values_AAPL)
close_median2 = statistics.median(close_values_AAPL)
volume_median2 = statistics.median(volume_values_AAPL)

open_mode2 = statistics.mode(open_values_AAPL) #calculates AAPL mode
high_mode2 = statistics.mode(high_values_AAPL)
low_mode2 = statistics.mode(low_values_AAPL)
close_mode2 = statistics.mode(close_values_AAPL)
volume_mode2 = statistics.mode(volume_values_AAPL)



print("Mean Value at Opening Time (9:30AM EST) - AAL:", open_mean1) #prints AAL mean
print("Mean Value at Closing Time (4:00PM EST) - AAL: ", close_mean1)
print("Mean Highest Value (during the day) - AAL: ", high_mean1)
print("Mean Lowest Value (during the day) - AAL", low_mean1)
print("Mean Volume - AAL: ", volume_mean1)

print("Median Value at Opening Time (9:30AM EST) - AAL:", open_median1) #prints AAL median
print("Median Value at Closing Time (4:00PM EST) - AAL: ", close_median1)
print("Median Highest Value (during the day) - AAL: ", high_median1)
print("Median Lowest Value (during the day) - AAL", low_median1)
print("Median Volume - AAL: ", volume_median1)

print("Mode Value at Opening Time (9:30AM EST) - AAL:", open_mode1) #prints AAL mode
print("Mode Value at Closing Time (4:00PM EST) - AAL: ", close_mode1)
print("Mode Highest Value (during the day) - AAL: ", high_mode1)
print("Mode Lowest Value (during the day) - AAL", low_mode1)
print("Mode Volume - AAL: ", volume_mode1)



print("Mean Value at Opening Time (9:30AM EST) - AAPL:", open_mean2) #prints AAPL mean
print("Mean Value at Closing Time (4:00PM EST) - AAPL: ", close_mean2)
print("Mean Highest Value (during the day) - AAPL: ", high_mean2)
print("Mean Lowest Value (during the day) - AAPL", low_mean2)
print("Mean Volume - AAPL: ", volume_mean2)

print("Median Value at Opening Time (9:30AM EST) - AAPL:", open_median2) #prints AAPL median
print("Median Value at Closing Time (4:00PM EST) - AAPL: ", close_median2)
print("Median Highest Value (during the day) - AAPL: ", high_median2)
print("Median Lowest Value (during the day) - AAPL", low_median2)
print("Median Volume - AAPL: ", volume_median2)

print("Mode Value at Opening Time (9:30AM EST) - AAPL:", open_mode2) #prints AAPL mode
print("Mode Value at Closing Time (4:00PM EST) - AAPL: ", close_mode2)
print("Mode Highest Value (during the day) - AAPL: ", high_mode2)
print("Mode Lowest Value (during the day) - AAPL", low_mode2)
print("Mode Volume - AAPL: ", volume_mode2)





date_index = header.index('date')

unique_dates_AAL = set() #creates sets to store unique dates for each of the 2 stocks in
unique_dates_AAPL = set()

with open('/Users/irisyura/Downloads/all_stocks_5yr.csv')as file:
    dataset = csv.reader(file)
    header = next(dataset)

    for lines in dataset:
        if lines[name_index] == AAL:
            unique_dates_AAL.add(lines[date_index]) #obtains the date in each row and adds it to the set -- this works because sets automatically ignore duplicates, so it identifies the number of unique dates
        elif lines[name_index] == AAPL:
            unique_dates_AAPL.add(lines[date_index])

print("Number of Unique Trading Days (over 5 years) - AAL: ", len(unique_dates_AAL))
print("Number of Unique Trading Days (over 5 years) - AAPL: ", len(unique_dates_AAPL))
