# lol




 




Statistical Data Analysis Lab Manual
 (B19CS5070)
III Year—5th Semester
Batch 2018-2022






SCHOOL OF COMPUTING
AND
INFORMATION TECHNOLOGY






CONTENTS
S.NO	PROGRAMS	Page No.
I	Course description	4
II	Course outcomes	4
III	Course objectives	4
IV	Guidelines to students	4
V	Lab requirements	5

PROGRAMS TO BE PRACTICED 
0	Anaconda is a free and open-source distribution of the Python and R programming languages for data science, machine learning applications, large-scale data processing, predictive analytics, etc.. Anaconda makes package management and deployment very simple. Download and Install of Anaconda Distribution of Python, understanding of Jupyter Notebook and various menu items in it, importing modules like Pandas, Numpy, SciPy etc. 	6
1.		Data frames in Python Pandas are excellent objects to handle tabular data from various sources. Write a python program to demonstrate creation of data frame using various formats of input data	7
2.		Any real time project involves data munging and data wrangling which involves selecting required rows and columns of data and manipulations on them. Write a python program to demonstrate following operations on rows and columns of a data frame:
a.	Selection   
b.	Insertion
c.	Deletion	14
3.		In order to explore the dataset and understand insights from it, the measures of central tendency play a crucial role, python has a strong set of functions that help you explore data. Write a python program to compute  descriptive statistics for measures of central tendency from given data: - Mean, Geometric Mean, Harmonic Mean, Median, and Mode	20
4.		Measuring descriptive statistics in data helps you decide what kind of processing to perform on it to gain useful information from data. Write a python program to compute  descriptive Statistics for Measures of Variability from given data- Variance, Standard Deviation, Skew, Percentiles	27
5.		Correlation is a statistical technique that can show whether and how strongly pairs of variables are related. For example, height and weight are related; taller people tend to be heavier than shorter people. Write a python program to compute measures of Correlation in given data - using Pearson, Spearman correlation coefficients.	36
6.		Data Visualization is visually representing the data using different plots/graphs/charts to find out the pattern, outliers, and relation between different attributes of a dataset. It is a graphical representation of data that helps human eye to detect patterns in data hence helps give a direction to data analysis tasks. Write a python program to plot following graphs using Matplotlib – Scatter plot, Box Plot, Bar Chart, Pie Chart	42
7.		Often a data analyst needs to combine data in a data frame by some criteria. This is done by providing a label to group data in the table. The pandas functions allow us to merge as well group data along rows and columns as per various criteria. Write a python program to demonstrate following operations on two data frames:
a.	Merging 
b.	GroupBy	48
8.		Hypothesis testing is a core concept in inferential statistics and a critical skill in the repertoire of a data scientist. The t-test is statistical hypothesis test in which the test statistic follows a Student's t-distribution under the null hypothesis. It is applied for normally distributed test statistic. Write a python program to demonstrate Hypothesis testing using Student’s T Test.	58
9.		Regression is a technique for searching relationships among variables in data. E.g. trying to understand relationship between salary and experience in years for employees in a data set containing employee information. Linear regression involves relation between one dependent and one independent variable. Write a python program to apply simple linear regression on data	61
10.		Multiple regression involves analysis of relation between one dependent and two or more independent variables. E.g. trying to establish relation between CGPA of a student as dependent on Attendance and SGPA. Write a python program to apply multiple linear regression on data	65
11.		Demonstrate following on Boston Dataset - Import every 50th row of Boston Housing dataset as a dataframe, then Import the Boston housing dataset, but while importing change the 'medv' (median house value) column so that values < 25 becomes ‘Low’ and > 25 becomes ‘High’.	67
12.		Project -Data analysis aims to find interesting insights from the data. Using the Boston Housing Dataset, apply the learnt concepts of data summarization, visualization, correlation and regression to derive your own interesting insights from the data. The results of this program may vary from student to student.	70
                                                           
Additional Programs
13.		Write a Python program to creat a numpy array using various input data in different formats.	
14.		Write a Python program to demonstrate indexing in numpy array	
15.		Write a Python program to demonstrate basic operations on single array	
16.		Write a Python program to demonstrate following operations using numpy
a.	Addition and Subtraction
b.	Square root
c.	Transpose	
17.		Write a Python program to demonstrate horizontal and vertical stacking in Python	
18.		Write a Python program to plot heat map of a data set	
19.		Write a Python program to demonstrate conditional deletion of columns and addition of a new column to a data frame.	
20.		Writa a Python program to demonstrate Classification operation on a dataset	
21.		Write a Python program to demonstrate K-Means clustering on a dataset	
22.		Write a Python program to demonstrate creation of a pandas dataframe from various input formats.	
 
I. COURSE DESCRIPTION

This laboratory course enables students to practice different statistical techniques used in data analysis pipeline using Python. We will explore the data analysis ecosystem of Python. Publicly available dataset will be downloaded. For the downloaded data set, we will see different data cleaning techniques like normalization and standardization, calculation of summary statistics, numerical and string operations on statistics. The aim of this course is to enable a student to manipulate data by application of statistical methods and derive interesting relationships and regularities among elements of data. This course will also introduce visualization tools like matplotlib and seaborn in Python. Visualization of data often helps to get a better understanding of the data. We will also cover few machine learning methods for making predictions about neew data. We will see the efficient libraries like NumPy, Pandas, Matplotlib, and SciPy available in Python for analysis of large amounts of data. The students are are expected to know how to program in any programming language. Knowledge of linear algebra, probability and statistics is a prerequisite for this lab. 

II. LAB OBJECTIVES

The objectives of this lab course are to enable a student:
1.	To write code for  reading from various data sources and writing results back
2.	To derive statistical measures from data
3.	To apply regression and correlation operations on data
4.	To generate plots from data 

III. LAB OUTCOMES
On successful completion of this course; the student will be able to:
CO1: Develop python code to read, write and manipulate data from files
CO2: Design python programs to compute statistical measures like mean, median, mode on a given data set.
CO3: Measure the regression coefficients among data values
CO4: Inspect data by plotting different kinds of graphs on it.

IV.GUIDELINES TO THE STUDENTS

1.	The students should have the basic knowledge of the Statistics and Probability Distributions
2.	This course is interrelated to theory taught in the class, please don’t miss both the class as well as labs.


V. LAB REQUIREMENTS


Following are the required hardware and software for this lab, which is available in the laboratory.
	Hardware: Desktop system or Virtual machine in a cloud with OS installed. Presently in the Lab, Pentium IV Processor having 1 GB RAM and 250 GB Hard Disk is available. Desktop systems are dual boot having Windows as well as Linux OS installed on them.
	Software: Python Interpreter , preferably the Anaconda Distribution of Python
This programming manual can be read simply as a text; however it is intended to be interactive. That is, you should be executing, modifying and using the programs that are presented herein.   

ANACONDA
Anaconda is a python distribution that provides you Python along with a large number of useful libraries and modules required for data analysis. 
Installation on Linux:
1.	Download latest version of Anaconda (Python 3.x)  from  https://www.anaconda.com/products/individual
2.	Download appropriate installer based on your operating system and follow the instructions.
For Windows : https://docs.anaconda.com/anaconda/install/windows/
On Linux: https://docs.anaconda.com/anaconda/install/linux/
On MacOS:  https://docs.anaconda.com/anaconda/install/mac-os/

Jupyter Notebook
The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more. Jupyter notebook comes bundled with your Anaconda installation.
Jupyter Notebook can also be installed separately. The instructions are here : https://jupyter.readthedocs.io/en/latest/install.html
Books and References:
1. McKinney, W. (2012). Python for data analysis: Data wrangling with Pandas, NumPy, and IPython. “O’ Reilly Media, Inc.".
2. Swaroop, C. H. (2003). A Byte of Python. Python Tutorial.
3. Jay L. Devore (2011). Probability and Statistics for Engineering and the Sciences. “Cengage Learning”. 
4. https://docs.python.org/3/library/statistics.html
5. https://scipy-lectures.org/packages/statistics/index.html



Program 0

0	Problem Statement 
Download and Install of Anaconda Distribution of Python,  understand how to use Jupyter Notebook and various menu items in it, learn how to import modules like Pandas, Numpy,SciPy
1	Student Learning Outcomes
After successfully following the steps listed in this section, the students will be able to :
a.	Download appropriate version of Anaconda and install it on their system
b.	Use the required functionality of Jupyter Notebook for executing the rest of the experiments
c.	Import the python modules
2	Description
ANACONDA
Anaconda is a python distribution that provides you Python along with a large number of useful libraries and modules required for data analysis. 
Installation on Linux:
1.	Download latest version of Anaconda (Python 3.x)  from  https://www.anaconda.com/products/individual
2.	Download appropriate installer based on your operating system and follow the instructions.
For Windows : https://docs.anaconda.com/anaconda/install/windows/
On Linux: https://docs.anaconda.com/anaconda/install/linux/
On MacOS:  https://docs.anaconda.com/anaconda/install/mac-os/

Jupyter Notebook
The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. Uses include: data cleaning and transformation, numerical simulation, statistical modeling, data visualization, machine learning, and much more. Jupyter notebook comes bundled with your Anaconda installation.
Jupyter Notebook can also be installed separately. The instructions are here : https://jupyter.readthedocs.io/en/latest/install.html
Importing Modules:

Modules are Python .py files that consist of Python code. Any Python file can be referenced as a module. A Python file called hello.py has the module name of hello that can be imported into other Python files or used on the Python command line interpreter.  Modules can define functions, classes, and variables that you can reference in other Python .py files or via the Python command line interpreter. In Python, modules are accessed by using the import statement. When you do this, you execute the code of the module, keeping the scopes of the definitions so that your current file(s) can make use of these. When Python imports a module called hello for example, the interpreter will first search for a built-in module called hello. If a built-in module is not found, the Python interpreter will then search for a file named hello.py in a list of directories that it receives from the sys.path variable.To make use of the functions in a module, you’ll need to import the module with in import statement. An import statement is made up of the import keyword along with the name of the module.
e.g. 
import pandas as pd

Above statement will load all the functionality defined in the module called ‘pandas’ into your environment so that you can use it easily in your program.
Program 1 
1	Problem Statement 
Write a python program to demonstrate creation of data frame using various formats of input data
2	Student Learning Outcomes
After successful completion of this lab program, the student shall be able to
•	Import required packages
•	Use various parameters to pandas DataFrame () function to create data frames.
•	Print the created data frames
3	Design of the Program 
3.1	Description
 

1.	A pandas DataFrame can be created using the following constructor:

pandas.DataFrame( data, index, columns, dtype, copy)

2.	A pandas DataFrame can be created using various inputs like −
Lists, dict,Series,Numpy ndarrays,Another DataFrame

3.2	Algorithm 
1.	Import pandas as pd
2.	Import numpy as np
3.	Supply various Python sequence objects containing data values to DataFrame() for creation of data frames
4.	Print the contents of Data frame
3.3	Python Code 
In this program, we will see the program 1of Statistical Data Analysis Lab for 5th Sem, CSE students.
Program 1: Write a python program to demonstrate following data pre-processing operations on a dataset.
In [2]:
#import the pandas library and aliasing as pd
#Creation of empty data frame
import pandas as pd
df = pd.DataFrame()
df
Out[2]:

In [3]:
# Creation of dataframe from a list of 5 numbers
import pandas as pd
data = [1,2,3,4,5]
df = pd.DataFrame(data)
df
Expected Output:

Out[3]:
	0
0	1
1	2
2	3
3	4
4	5
In [4]:
# Creation of dataframe from a list of lists containing numbers
import pandas as pd
data = [[1,2,3],[4,5]]
# When we don't specify column headings, python allocates numeric column headings and row indices, 
#these indices begin at 0
df = pd.DataFrame(data)
df
Expected Output:

Out[4]:
	0	1	2
0	1	2	3.0
1	4	5	NaN
In [5]:
#Create dataframe from a list of lists
import pandas as pd
data = [['Ram',10],['Hari',12],['Madhav',13]]
# The parameter columns in DataFrame() helps you to specify names of columns or column headings
df = pd.DataFrame(data,columns=['Name','Age'])
df
Expected Output:

Out[5]:
	Name	Age
0	Ram	10
1	Hari	12
2	Madhav	13
In [6]:
# Creating dataframe  from a list of lists with data type specified
import pandas as pd
data = [['Alex',10],['Bob',12],['Clarke',13]]
#Here we have specified data, column headings and the datatype that should be applied to Age column
df = pd.DataFrame(data,columns=['Name','Age'],dtype=float)
df
Expected Output:

Out[6]:
	Name	Age
0	Alex	10.0
1	Bob	12.0
2	Clarke	13.0
In [7]:
#Creating a data frame from dictionary and a list
import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data)
df
Expected Output:
Out[7]:
	Name	Age
0	Tom	28
1	Jack	34
2	Steve	29
3	Ricky	42
In [8]:
#Creating a data frame from dictionary, with row index specified
import pandas as pd
data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
df = pd.DataFrame(data, index=['rank1','rank2','rank3','rank4'])
df
Expected Output:
Out[8]:
	Name	Age
rank1	Tom	28
rank2	Jack	34
rank3	Steve	29
rank4	Ricky	42
In [9]:
#Creating a data frame From list of dictionaries
import pandas as pd
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
df
Expected Output:
Out[9]:
	a	b	c
0	1	2	NaN
1	5	10	20.0
In [10]:
#Creating a data frame from a List of dictionaries with row index specified
import pandas as pd
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data, index=['first', 'second'])
df
Expected Output:
Out[10]:
	a	b	c
first	1	2	NaN
second	5	10	20.0

In [11]:
#Creating a data frame using a list of dictionaries
import pandas as pd
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]

#With two column indices, values same as dictionary keys
df1 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b'])

#With two column indices with one index with other name
df2 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b1'])
df1
df2
Expected Output:
Out[11]:
	a	b1
first	1	NaN
second	5	NaN
In [12]:
#Creating a data frame using a dictionary and series
import pandas as pd

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
df
Expected Output:
Out[12]:
	one	two
a	1.0	1
b	2.0	2
c	3.0	3
d	NaN	4
















Program 2 
1	Problem Statement 

Write a python program to demonstrate following operations on rows and columns of a data frame:
a.	Selection   
b.	Insertion
c.	Deletion
2	Student Learning Outcomes
After completion of this lab experiment, the students shall be able to :
a.	Create data frame from suitable data
b.	Use indexing to select rows and columns from data frame
c.	Insert new rows and new columns in data frame
d.	Delete specified rows and columns form data frame
3	Design of the Program 
3.1	Description:

We can select required rows and column from a data frame using indexing and slicing operators. These operators are [], loc[] and iloc[]


Dataframe.[ ] ; This function also known as indexing operator
Dataframe.loc[ ] : This function is used for labels.
Dataframe.iloc[ ] : This function is used for positions or integer based
Dataframe.ix[] : This function is used for both label and integer based
Collectively, they are called the indexers. These are by far the most common ways to index data. These are four function which help in getting the elements, rows, and columns from a DataFrame.
 
Indexing a Dataframe using indexing operator [] :
Indexing operator is used to refer to the square brackets following an object. The .loc and .iloc indexers also use the indexing operator to make selections. 

3.2	Algorithm
1.	Import required modules
2.	Load a dataset or create the data by specifying values in suitable python object e.g. list, array, data frame or series.
3.	Use indexing operator to select appropriate columns and rows. 
4.	Use correct syntax to insert, delete data in columns and rows.
3.3	Python Code  

In this program, we will see how to access one or more rows or columns of a data frame, how to add new rows and columns and how to delete specified rows and columns from data frame.

In [1]:
#Selecting and printing  one column of a data frame created using a dictionary 
import pandas as pd

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print (df ['one'])
a    1.0
b    2.0
c    3.0
d    NaN
Name: one, dtype: float64

In [2]:
#Adding a new data column to a data frame, new column is created using Series()
import pandas as pd

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)

# Adding a new column to an existing DataFrame object with column label by passing new series

print ("Adding a new column by passing as Series:")
df['three']=pd.Series([10,20,30],index=['a','b','c'])
print(df)

print ("Adding a new column using the existing columns in DataFrame:")
df['four']=df['one']+df['three']

df
Adding a new column by passing as Series:
   one  two  three
a  1.0    1   10.0
b  2.0    2   20.0
c  3.0    3   30.0
d  NaN    4    NaN
Adding a new column using the existing columns in DataFrame:
Expected Output:

Out[2]:
	one	two	three	four
a	1.0	1	10.0	11.0
b	2.0	2	20.0	22.0
c	3.0	3	30.0	33.0
d	NaN	4	NaN	NaN
In [3]:
# we will delete a column using del function and dusing pop() function
import pandas as pd

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd']), 
   'three' : pd.Series([10,20,30], index=['a','b','c'])}

df = pd.DataFrame(d)
print ("Our dataframe is:")
print (df)

# using del function
print ("Deleting the first column using DEL function:")
del df['one']
print (df)

# using pop function
print ("Deleting another column using POP function:")
df.pop('two')
df
Expected Output:

Our dataframe is:
   one  two  three
a  1.0    1   10.0
b  2.0    2   20.0
c  3.0    3   30.0
d  NaN    4    NaN
Deleting the first column using DEL function:
   two  three
a    1   10.0
b    2   20.0
c    3   30.0
d    4    NaN
Deleting another column using POP function:
Expected Output:

Out[3]:
	three
a	10.0
b	20.0
c	30.0
d	NaN
In [4]:
#Accessing a row using .loc()
import pandas as pd

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print (df.loc['b'])
one    2.0
two    2.0
Name: b, dtype: float64

In [5]:
#Accessing a row using iloc[]
import pandas as pd

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print (df.iloc[2])
one    3.0
two    3.0
Name: c, dtype: float64
In [6]:
#Slicing a Data Frame created from Series
import pandas as pd

d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']), 
   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print (df[2:4])
Expected Output:

   one  two
c  3.0    3
d  NaN    4
In [7]:
# Appending two data frames using append() function
import pandas as pd

df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])

df = df.append(df2)
df
Expected Output:
Out[7]:
	a	b
0	1	2
1	3	4
0	5	6
1	7	8
In [8]:
#Deleting a Row using drop(0) function
import pandas as pd
df = pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])

df = df.append(df2)
# Drop rows with label 0
df = df.drop(0)
df
Expected Output:
Out[8]:
	a	b
1	3	4
1	7	8













Program 3 
1	Problem Statement 

Write a python program to compute descriptive statistics for measures of central tendency from given data: - Mean, Geometric Mean, Harmonic Mean, Median, and Mode
2	Student Learning Outcomes
After completion of this lab experiment, the student shall be able to :
a.	Decide which kind of measure to apply depending of data type
b.	Apply the pandas functions to compute the measures from given data  
3	Design of the Program 
3.1	Description:
What is central tendency? 
The central tendency is a single number that represents the most common value for a list of numbers. There are many ways to calculate the central tendency for a data sample, such as the mean which is average value calculated from the values, the mode, which is the most common value in the data distribution, or the  median, which is the middle value if all values in the data sample were ordered.
Three common types of mean calculations that you may encounter are the arithmetic mean, the geometric mean, and the harmonic mean. 

Arithmetic Mean
The arithmetic mean is calculated as the sum of the values divided by the total number of values, referred to as N. 

•	Arithmetic Mean = (x1 + x2 + … + xN) / N
A more convenient way to calculate the arithmetic mean is to calculate the sum of the values and to multiply it by the reciprocal of the number of values (1 over N); for example:
•	Arithmetic Mean = (1/N) * (x1 + x2 + … + xN)
The arithmetic mean is appropriate when all values in the data sample have the same data type. The values of arithmetic mean can be positive, negative, or zero. The arithmetic mean is useful in machine learning when summarizing a variable, e.g. reporting the most likely value. The arithmetic mean can be calculated using the mean() NumPy function.
Geometric Mean
The geometric mean is calculated as the N-th root of the product of all values, where N is the number of values.
•	Geometric Mean = N-root(x1 * x2 * … * xN)
For example, if the data contains only two values, the square root of the product of the two values is the geometric mean. For three values, the cube-root is used, and so on.
The geometric mean is appropriate when the data contains values with different units of measure, e.g. some measure are height, some are dollars, some are miles, etc. The geometric mean does not accept negative or zero values, e.g. all values must be positive.  The geometric mean can be calculated using the gmean() SciPy function.
Harmonic Mean
The harmonic mean is calculated as the number of values N divided by the sum of the reciprocal of the values (1 over each value).
•	Harmonic Mean = N / (1/x1 + 1/x2 + … + 1/xN)
If there are just two values (x1 and x2), a simplified calculation of the harmonic mean can be calculated as:
•	Harmonic Mean = (2 * x1 * x2) / (x1 + x2)
The harmonic mean is the appropriate mean if the data is comprised of rates.
Recall that a rate is the ratio between two quantities with different measures, e.g. speed, acceleration, frequency, etc. The harmonic mean does not take rates with a negative or zero value, e.g. all rates must be positive. The harmonic mean can be calculated using the hmean() SciPy function.

Each mean is appropriate for different types of data; for example:
•	If values have the same units: Use the arithmetic mean.
•	If values have differing units: Use the geometric mean.
•	If values are rates: Use the harmonic mean.
The exceptions are if the data contains negative or zero values, then the geometric and harmonic means cannot be used directly.
3.2	Algorithm
        	
1.	Import required modules
2.	Load a dataset or create the data by specifying values in suitable python object e.g. list, array, data frame or series.
3.	Apply functions for calculation of mean, median and mode.
3.3	Python Code  

# example of calculating the arithmetic mean
from numpy import mean
# define the dataset
data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# calculate the mean
result = mean(data)
print('Arithmetic Mean: %.3f' % result)
Expected Output:
Arithmetic Mean: 4.500




In [10]:
# example of calculating the geometric mean
from scipy.stats import gmean
# define the dataset
data = [1, 2, 3, 40, 50, 60, 0.7, 0.88, 0.9, 1000]
# calculate the mean
result = gmean(data)
print('Geometric Mean: %.3f' % result)
Expected Output:
Geometric Mean: 7.246

In [11]:
# example of calculating the harmonic mean
from scipy.stats import hmean
# define the dataset
data = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99]
# calculate the mean
result = hmean(data)
print('Harmonic Mean: %.3f' % result)
Expected Output:
Harmonic Mean: 0.350

We can compute Median using dataframe.median() function Lets use the dataframe.median() function to find the median over the index axis By default, this function will return the median over the index axis or row axis You can specify axis = 1 if you want median over column axis
In [2]:
# importing pandas as pd 
import pandas as pd 

# Creating the dataframe 
df = pd.DataFrame({"A":[12, 4, 5, 44, 1], 
                "B":[5, 2, 54, 3, 2], 
                "C":[20, 16, 7, 3, 8], 
                "D":[14, 3, 17, 2, 6]}) 

# Print the dataframe 
df 
Expected Output:
Out[2]:
	A	B	C	D
0	12	5	20	14
1	4	2	16	3
2	5	54	7	17
3	44	3	3	2
4	1	2	8	6
In [3]:
#Calculate the median over row axis
print("The median of the data frame on the row axis is:")
df.median(axis = 0) 
The median of the data frame on the row axis is:
Expected Output:
Out[3]:
A    5.0
B    3.0
C    8.0
D    6.0
dtype: float64
Calculate the median over column axis
In [5]:
print("The median of the data frame on the column axis is:")
df.median(axis = 1)
The median of the data frame on the column axis is:
Expected Output:
Out[5]:
0    13.0
1     3.5
2    12.0
3     3.0
4     4.0
dtype: float64
In [1]:
# Let us compute median when data frame has missing values 
import pandas as pd 
# Creating the dataframe 
df = pd.DataFrame({"A":[12, 4, 5, None, 1], 
                "B":[7, 2, 54, 3, None], 
                "C":[20, 16, 11, 3, 8], 
                "D":[14, 3, None, 2, 6]}) 

# Print the dataframe 
df 
Expected Output:
Out[1]:
	A	B	C	D
0	12.0	7.0	20	14.0
1	4.0	2.0	16	3.0
2	5.0	54.0	11	NaN
3	NaN	3.0	3	2.0
4	1.0	NaN	8	6.0
In [8]:
# Compute median over columns,skip the missing or not available (Na) values while finding the median 
print("The median of the data frame on the column axis is (we have skipped the missing values):")
df.median(axis = 1, skipna = True) 
The median of the data frame on the column axis is (we have skipped the missing values):
Expected Output:
Out[8]:
0    13.0
1     3.5
2    12.0
3     3.0
4     4.0
dtype: float64
Computing MODE
In [ ]:
#mode
# importing pandas as pd 
import pandas as pd 

# Creating the dataframe 
df=pd.DataFrame({"A":[14,4,5,4,1], 
                "B":[5,2,54,3,2], 
                "C":[20,20,7,3,8], 
                "D":[14,3,6,2,6]}) 

In [9]:
# find mode of dataframe on row axis
print("The mode of the data frame on the row axis is:")
df.mode() 
Expected Output:
The mode of the data frame on the row axis is:
Out[9]:
	A	B	C	D
0	1	2.0	3	2
1	4	NaN	7	3
2	5	NaN	8	6
3	12	NaN	16	14
4	44	NaN	20	17
In [10]:
# axis = 1 indicates over the column axis 
print("The mode of the data frame on the column axis is:")

df.mode(axis = 1) 
The mode of the data frame on the column axis is:
Expected Output:
Out[10]:
	0	1	2	3
0	5.0	12.0	14.0	20.0
1	2.0	3.0	4.0	16.0
2	5.0	7.0	17.0	54.0
3	3.0	NaN	NaN	NaN
4	1.0	2.0	6.0	8.0






Program 4
Problem Statement 
Write a python program to compute  descriptive Statistics for Measures of Variability from given data- Variance, Standard Deviation, Skew, Quantile
Student Learning Outcomes
After successful completion of this program, the student will be able to:
4.	Decide on the appropriate functions to use for calculating the required measure
5.	Use the correct syntax of the function to get the expected results
3	Design of the Program 
3.1	Description
   
Variance : 
The function var() – in python pandas is used to calculate variance of a given set of numbers, Variance of a data frame, Variance of column or column wise variance in pandas python  and Variance of rows or row wise variance in pandas python.  It is present in the package “statistics” in Python.
Syntax of variance Function in python
DataFrame.var(axis=None, skipna=None, level=None, ddof=1, numeric_only=None)
axis : {rows (0), columns (1)}
skipna : Exclude NA/null values when computing the result
level : If the axis is a MultiIndex (hierarchical), count along a particular level, collapsing into a Series
ddof: Delta Degrees of Freedom. The divisor used in calculations is N – ddof, where N represents the number of elements.
numeric_only: Include only float, int, boolean columns. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.

Standard Deviation:
Pandas dataframe.std () function return sample standard deviation over requested axis. By default the standard deviations are normalized by N-1. It is a measure that is used to quantify the amount of variation or dispersion of a set of data values.

Syntax : 
DataFrame.std(axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs)
Parameters: 
axis: {index (0), columns (1)}
skipna : Exclude NA/null values. If an entire row/column is NA, the result will be NA
level: If the axis is a MultiIndex (hierarchical), count along a particular level, collapsing into a Series
ddof: Delta Degrees of Freedom. The divisor used in calculations is N – ddof, where N represents the number of elements.
numeric_only: Include only float, int, boolean columns. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
Return : std : Series or DataFrame (if level specified)

Skew: 
Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean.
Pandas dataframe.skew() function return unbiased skew over requested axis Normalized by N-1.

DataFrame.skew(axis=None, skipna=None, level=None, numeric_only=None, **kwargs)
Parameters :
axis : {index (0), columns (1)}
skipna : Exclude NA/null values when computing the result.
level : If the axis is a MultiIndex (hierarchical), count along a particular level, collapsing into a Series
numeric_only : Include only float, int, boolean columns. If None, will attempt to use everything, then use only numeric data. Not implemented for Series.
Return : skew : Series or DataFrame (if level specified)

Quantile()
Pandas dataframe.quantile() function return values at the given quantile over requested axis, a numpy.percentile.
Note: In each of any set of values of a variate which divide a frequency distribution into equal groups, each containing the same fraction of the total population.

Syntax: DataFrame.quantile(q=0.5, axis=0, numeric_only=True, interpolation=’linear’)
Parameters: 
q : float or array-like, default 0.5 (50% quantile). 0 <= q <= 1, the quantile(s) to compute
axis : [{0, 1, ‘index’, ‘columns’} (default 0)] 0 or ‘index’ for row-wise, 1 or ‘columns’ for column-wise
numeric_only : If False, the quantile of datetime and timedelta data will be computed as well
interpolatoin : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
Returns: quantiles: Series or DataFrame
-> If q is an array, a DataFrame will be returned where the index is q, the columns are the columns of self, and the values are the quantiles.
-> If q is a float, a Series will be returned where the index is the columns of self and the values are the quantiles.


3.2	Algorithm 

3.3	Python Code 
Variance
In [1]:
#Computing Variance in a dataframe
import pandas as pd
import numpy as np
 
#Create a DataFrame
d = {
    'Name':['Alisa','Bobby','Cathrine','Madonna','Rocky','Sebastian','Jaqluine',
   'Rahul','David','Andrew','Ajay','Teresa'],
   'Score1':[62,47,55,74,31,77,85,63,42,32,71,57],
   'Score2':[89,87,67,55,47,72,76,79,44,92,99,69],
   'Score3':[56,86,77,45,73,62,74,89,71,67,97,68]}

df = pd.DataFrame(d)
In [2]:
df
Expected Output:
Out[2]:
	Name	Score1	Score2	Score3
0	Alisa	62	89	56
1	Bobby	47	87	86
2	Cathrine	55	67	77
3	Madonna	74	55	45
4	Rocky	31	47	73
5	Sebastian	77	72	62
6	Jaqluine	85	76	74
7	Rahul	63	79	89
8	David	42	44	71
9	Andrew	32	92	67
10	Ajay	71	99	97
11	Teresa	57	69	68
In [3]:
# variance of the dataframe for each column 
df.var()
Expected Output:
Out[3]:
Score1    304.363636
Score2    311.636364
Score3    206.083333
dtype: float64
In [6]:
# column variance of the dataframe for each row
#Notice that axis =0, which is unlike the value used in previous programs, where axis =0 operated on rows.
df.var(axis=1)
Expected Output:
Out[6]:
0     309.000000
1     520.333333
2     121.333333
3     217.000000
4     449.333333
5      58.333333
6      34.333333
7     172.000000
8     262.333333
9     908.333333
10    244.000000
11     44.333333
dtype: float64
In [7]:
# variance of the specific column can be computed by accessing that column by name using loc[] on data frame
df.loc[:,"Score1"].var()
Expected Output:
Out[7]:
304.3636363636364
Standard Deviation
In [8]:
# Computing Standard Deviation
import pandas as pd  
import numpy as np  
  
#Create a DataFrame  
info = {  
    'Name':['Parker','Smith','John','William'],  
   'sub1_Marks':[52,38,42,37],  
   'sub2_Marks':[41,35,29,36]}   
data = pd.DataFrame(info)  
Expected Output:
Out[8]:
sub1_Marks    6.849574
sub2_Marks    4.924429
dtype: float64
In [10]:
data
Expected Output:
Out[10]:
	Name	sub1_Marks	sub2_Marks
0	Parker	52	41
1	Smith	38	35
2	John	42	29
3	William	37	36
In [11]:
# standard deviation of the dataframe for each column 
data.std()  
Expected Output:
Out[11]:
sub1_Marks    6.849574
sub2_Marks    4.924429
dtype: float64
In [9]:
# Standard deviation of dataframe for each row
data.std(axis=1)
Expected Output:
Out[9]:
0    7.778175
1    2.121320
2    9.192388
3    0.707107
dtype: float64
SKEW
A skewness value of 0 in the output denotes a symmetrical distribution of values in row 1. A negative skewness value in the output indicates an asymmetry in the distribution corresponding to row 2 and the tail is larger towards the left hand side of the distribution. A positive skewness value in the output indicates an asymmetry in the distribution corresponding to row 3 and the tail is larger towards the right hand side of the distribution.
In [12]:
import pandas as pd

dataVal = [(10,20,30,40,50,60,70),
           (10,10,40,40,50,60,70),
           (10,20,30,50,50,60,80)]

dataFrame = pd.DataFrame(data=dataVal);
skewValue = dataFrame.skew(axis=1)
print("DataFrame:")
print(dataFrame)

print("Skew:")
print(skewValue)

Expected Output:
DataFrame:
    0   1   2   3   4   5   6
0  10  20  30  40  50  60  70
1  10  10  40  40  50  60  70
2  10  20  30  50  50  60  80
Skew:
0    0.000000
1   -0.340998
2    0.121467
dtype: float64
Quantile
In [13]:
# importing pandas as pd 
import pandas as pd 

# Creating the dataframe 
df = pd.DataFrame({"A":[1, 5, 3, 4, 2], 
                    "B":[3, 2, 4, 3, 4], 
                    "C":[2, 2, 7, 3, 4], 
                    "D":[4, 3, 6, 12, 7]}) 

# Print the dataframe 
df 
Expected Output:
Out[13]:
	A	B	C	D
0	1	3	2	4
1	5	2	2	3
2	3	4	7	6
3	4	3	3	12
4	2	4	4	7
In [16]:
# Compute 0.1, 0.25, 0.5 ,0.75 i.e. 4 quantiles of a data frame for each row
import pandas as pd 
# Creating the dataframe 
df = pd.DataFrame({"A":[1, 5, 3, 4, 2], 
                "B":[3, 2, 4, 3, 4], 
                "C":[2, 2, 7, 3, 4], 
                "D":[4, 3, 6, 12, 7]}) 

# using quantile() function to 
# find the quantiles over the index axis 
df.quantile([.1, .25, .5, .75], axis = 0) 
Expected Output:
Out[16]:
	A	B	C	D
0.10	1.4	2.4	2.0	3.4
0.25	2.0	3.0	2.0	4.0
0.50	3.0	3.0	3.0	6.0
0.75	4.0	4.0	4.0	7.0


Program 5 
1	Problem Statement 
 Write a python program to compute  measures of Correlation in given data - using Pearson, Spearman correlation coefficients
2	Student Learning Outcomes
After successful completion of this program, a student will be able to :
1.	Identify the correct python functions to use for computing correlation among data items
2.	Apply the functions correctly and compute the correlation coefficients.
   3	Design of the Program 
3.1	Description

Correlation is the relationship between two or more variables in a DataSet. A variable always has a correlation of 1 with itself. Pandas dataframe.corr() is used to find the pairwise correlation of all columns in the dataframe. Any na values are automatically excluded. For any non-numeric data type columns in the dataframe it is ignored.
yntax: DataFrame.corr(self, method=’pearson’, min_periods=1)
Parameters:
method :
pearson : standard correlation coefficient
kendall : Kendall Tau correlation coefficient
spearman : Spearman rank correlation
min_periods : Minimum number of observations required per pair of columns to have a valid result. Currently only available for pearson and spearman correlation
Returns: count :y : DataFrame
The correlation of a variable with itself is 1.

 
There are three important types of correlation coefficients:
•	Peason Correlation 
•	Spearman’s Correlation
•	Kendall’s Correlation
3.2	Algorithm 

Steps:
1.	Read CSV file into dataframe
2.	Compute Pearson Correlation Coefficient using corr() function with method = “pearson” parameter
3.	Compute Kendall Correlation Coefficient using corr() function with method = “Kendall” parameter
4.	Compute Spearman Correlation Coefficient using corr() function with method = “Spearman” parameter
5.	Create two data frames from data stores as dictionaries
6.	Compute their columnwise and row wise correlation using corr() with axis parameter at 0 and 1 respectively.
3.3	Python Code 
In the first part of program we use corr() function to find the correlation among the columns in the a dataframe created from iris.csv file using Pearson, Spearman and kendall correlation coefficients.
In [ ]:
# importing pandas as pd 
import pandas as pd 

# Making data frame from the csv file 
df = pd.read_csv("iris.csv") 
In [9]:
# Printing the first 10 rows of the data frame for visualization 
df[:10]
Out[9]:
	sepal_length	sepal_width	petal_length	petal_width	species
0	5.1	3.5	1.4	0.2	setosa
1	4.9	3.0	1.4	0.2	setosa
2	4.7	3.2	1.3	0.2	setosa
3	4.6	3.1	1.5	0.2	setosa
4	5.0	3.6	1.4	0.2	setosa
5	5.4	3.9	1.7	0.4	setosa
6	4.6	3.4	1.4	0.3	setosa
7	5.0	3.4	1.5	0.2	setosa
8	4.4	2.9	1.4	0.2	setosa
9	4.9	3.1	1.5	0.1	setosa
In [10]:
# To find the correlation among 
# the columns using pearson method 
df.corr(method ='pearson') 
Out[10]:
	sepal_length	sepal_width	petal_length	petal_width
sepal_length	1.000000	-0.109369	0.871754	0.817954
sepal_width	-0.109369	1.000000	-0.420516	-0.356544
petal_length	0.871754	-0.420516	1.000000	0.962757
petal_width	0.817954	-0.356544	0.962757	1.000000
In [11]:
# To find the correlation among 
# the columns using pearson method 
df.corr(method ='spearman') 
Out[11]:
	sepal_length	sepal_width	petal_length	petal_width
sepal_length	1.000000	-0.159457	0.881386	0.834421
sepal_width	-0.159457	1.000000	-0.303421	-0.277511
petal_length	0.881386	-0.303421	1.000000	0.936003
petal_width	0.834421	-0.277511	0.936003	1.000000
In [12]:
# To find the correlation among 
# the columns using pearson method 
df.corr(method ='kendall') 
Out[12]:
	sepal_length	sepal_width	petal_length	petal_width
sepal_length	1.000000	-0.072112	0.717624	0.654960
sepal_width	-0.072112	1.000000	-0.182391	-0.146988
petal_length	0.717624	-0.182391	1.000000	0.803014
petal_width	0.654960	-0.146988	0.803014	1.000000
In this part of the program , we create two different data frames from dictionaries and compute their mutual corration using corrwith() function
In [14]:
# importing pandas as pd 
import pandas as pd 

# Creating the first dataframe 
df1 = pd.DataFrame({"A":[1, 5, 7, 8], 
					"B":[5, 8, 4, 3], 
					"C":[10, 4, 9, 3]}) 

# Creating the second dataframe 
df2 = pd.DataFrame({"A":[5, 3, 6, 4], 
					"B":[11, 2, 4, 3], 
					"C":[4, 3, 8, 5]}) 

# Print the first dataframe 
print(df1, "\n") 

# Print the second dataframe 
print(df2) 
   A  B   C
0  1  5  10
1  5  8   4
2  7  4   9
3  8  3   3 

   A   B  C
0  5  11  4
1  3   2  3
2  6   4  8
3  4   3  5
In [15]:
# To find the correlation among the 
# columns of df1 and df2 along the column axis 
df1.corrwith(df2, axis = 0) 
Out[15]:
A   -0.041703
B   -0.151186
C    0.395437
dtype: float64
In [16]:
#ROW WISE CORRELATION

df1.corrwith(df2, axis = 1) 
Out[16]:
0   -0.195254
1   -0.970725
2    0.993399
3    0.000000
dtype: float64


Program 6
1	Problem Statement 

Write a python program to plot following graphs using Matplotlib – Scatter plot, Box Plot, Bar Chart, Pie Chart
2	Student Learning Outcomes
After successful completion of this lab, the student will able to

•	Import the libraries required for plotting
•	Use appropriate plot function
•	Label the axis of plots
•	Decide which type of plot to use depending on data and task
3	Design of the Program 
3.1	Description
Python has a library called Matplotlib which provides build in functions for drawing various kinds of plots. This is a library for 2-dimensional plotting with Python. Some plots it will let us build are: Plots, Histograms, Error charts, Bar charts, Scatter Plots etc. It has a pyplot interface. This holds command-like functions that let us alter a figure.  Some features of Python Plot supports- Font properties, Axes properties, Line styles.  
Matplot has a built-in function to create scatterplots called scatter(). A scatter plot is a type of plot that shows the data as a collection of points. The position of a point depends on its two-dimensional value, where each value is a position on either the horizontal or vertical dimension, A boxplot is a standardized way of displaying the distribution of data based on a five number summary (“minimum”, first quartile (Q1), median, third quartile (Q3), and “maximum”). It can tell you about your outliers and what their values are. It can also tell you if your data is symmetrical, how tightly your data is grouped, and if and how your data is skewed. A bar chart or bar graph is a chart or graph that presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent. The bars can be plotted vertically or horizontally. A vertical bar chart is sometimes called a column chart a Pie Chart is used to show parts to the whole, and often a % share.

 

3.2	Algorithm
1.	Import the required libraries
2.	Load the data into a data frame
3.	Decide which type of plot to draw
4.	Select the columns to plot on x and y axis
5.	Plot the data
6.	Plot appropriate labels for x and y axis	
3.3	Python Code 
#Scatter Plot
import numpy as np
import matplotlib.pyplot as plt
# Create data
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
# Plot
plt.scatter(x, y)
plt.title('Scatter plot')
plt.xlabel('This is x axis')
plt.ylabel('This is y axis')
plt.show()
 
In [57]:
# Scatter plot  of sine function
x=np.linspace(0,10,30)
from scipy import sin
y=np.sin(x)
plt.plot(x,y,'o',color='purple')
Out[57]:
[<matplotlib.lines.Line2D at 0x2dcd7c9cec8>]
In [58]:
# Scatter plot with more customized marker style
plt.plot(x,y,'-p',color='green',
     markersize=15,linewidth=4,
     markerfacecolor='white',
     markeredgecolor='gray',
     markeredgewidth=1)
Out[58]:
[<matplotlib.lines.Line2D at 0x2dcd9624608>]
 
In [59]:
#Box Plot
import matplotlib.pyplot as plt
 
value1 = [82,76,24,40,67,62,75,78,71,32,98,89,78,67,72,82,87,66,56,52]
value2=[62,5,91,25,36,32,96,95,3,90,95,32,27,55,100,15,71,11,37,21]
value3=[23,89,12,78,72,89,25,69,68,86,19,49,15,16,16,75,65,31,25,52]
value4=[59,73,70,16,81,61,88,98,10,87,29,72,16,23,72,88,78,99,75,30]
box_plot_data=[value1,value2,value3,value4]
plt.boxplot(box_plot_data)
plt.show()
 
In [38]:
#Bar Chart
import matplotlib.pyplot as plt
plt.bar([1,3,5,7,9],[5,2,7,8,2], label="Bar from first data set")
plt.bar([2,4,6,8,10],[8,6,2,5,6], label="Bar from 2nd data set", color='g')
plt.legend()
plt.xlabel('bar number')
plt.ylabel('bar height')
plt.title('Example of a bar chart')
plt.show()
 
In [39]:
#Bar Chart
import matplotlib.pyplot as plt
plt.bar([1,2,3,4,5],[100,190,270,350,430], label="Apples")
plt.bar([1,2,3,4,5],[50,90,150,200,260], label="Oranges", color='g')
plt.legend()
plt.xlabel('Weight in Kg')
plt.ylabel('Cost in Rupees')
plt.title('Cost of apples and oranges')
plt.show()
 
In [43]:
# Pie Chart
import matplotlib.pyplot as plt
slices = [7,2,2,13]
activities = ['sleeping','eating','working','playing']
cols = ['c','m','r','b']
plt.pie(slices,
        labels=activities,
        colors=cols,
        startangle=90,
        shadow= True,
        explode=(0,0.1,0,0),
        autopct='%1.1f%%')
plt.title('This is a pie chart')
plt.show()
 
In [ ]:
 










Program 7
1	Problem Statement 

. Write a python program to demonstrate following operations on two data frames:
a.	Merge
b.	Join
c.	GroupBy
2	Student Learning Outcomes
After successful completion of this lab, the student will able to

•	Load two or more data sets in to data frame or create dataframe 
•	Apply appropriate function to perform merging of data frames
•	Apply group by operation on required rows of the data frames
3	Design of the Program 
3.1	Description
	Pandas has a function called merge() that merges data frames or series objects with database style join  based on columns or rows (indexes). If joining columns on columns, the indexes are ignored, when joining indexes on indexes or indexes on column the index will be passed on. 
Pandas join() function joins columns with other  DataFrame either on row ( index) or on a key column. It can efficiently join multiple DataFrame objects by index at once by passing a list
Pandas groupby() function involves some combination of splitting the object, applying a function, and combining the results. This can be used to group large amounts of data and compute operations on these groups.
Parameters taken by groupby(): 

1.by -> mapping, function, label, or list of labels
Used to determine the groups for the groupby. If by is a function, it’s called on each value of the object’s index. If a dict or Series is passed, the Series or dict VALUES will be used to determine the groups (the Series’ values are first aligned; see .align() method). If an ndarray is passed, the values are used as-is determine the groups. A label or list of labels may be passed to group by the columns in self. Notice that a tuple is interpreted as a (single) key.

2.axis-> {0 or ‘index’, 1 or ‘columns’}, default 0
Split along rows (0) or columns (1).

3.level->int, level name, or sequence of such, default None
If the axis is a MultiIndex (hierarchical), group by a particular level or levels.

4.as_index->bool, default True
For aggregated output, return object with group labels as the index. Only relevant for DataFrame input. as_index=False is effectively “SQL-style” grouped output.

5.sort->bool, default True
Sort group keys. Get better performance by turning this off. Note this does not influence the order of observations within each group. Groupby preserves the order of rows within each group.

6.group_keys->bool, default True
When calling apply, add group keys to index to identify pieces.

7.squeeze->bool, default False
Reduce the dimensionality of the return type if possible, otherwise return a consistent type.

8.observed->bool, default False
This only applies if any of the groupers are Categoricals. If True: only show observed values for categorical groupers. If False: show all values for categorical groupers.

Returns
DataFrameGroupBy ->Returns a groupby object that contains information about the groups

3.2	Algorithm
	1.	Import the required libraries
2.	Load or create data frame objects with data
3.	Use correct form of merge() to demonstrate the required merging operation
4.	Use correct form of join() to demonstrate the required joining operation
5.	Use correct form of groupby() to demonstrate the required grouping operation on data frames.
3.3	Python Code 
#Merge
import pandas as pd
df1 = pd.DataFrame({'key1': ['foo', 'bar', 'baz', 'foo'],
                    'value': [1, 2, 3, 5]})
df2 = pd.DataFrame({'key2': ['foo', 'bar', 'baz', 'foo'],
                    'value': [5, 6, 7, 8]})
In [6]:
df1
Out[6]:
	key1	value
0	foo	1
1	bar	2
2	baz	3
3	foo	5
In [7]:
df2
Out[7]:
	key2	value
0	foo	5
1	bar	6
2	baz	7
3	foo	8
In [8]:
#Merge df1 and df2 on the lkey and rkey columns. 
#The value columns have the default suffixes, _x and _y, appended.

df1.merge(df2, left_on='key1', right_on='key2')
Out[8]:
	key1	value_x	key2	value_y
0	foo	1	foo	5
1	foo	1	foo	8
2	foo	5	foo	5
3	foo	5	foo	8
4	bar	2	bar	6
5	baz	3	baz	7
In [9]:
#Merge DataFrames df1 and df2 with specified left and right suffixes appended to any overlapping columns.

df1.merge(df2, left_on='key1', right_on='key2',
          suffixes=('_left', '_right'))
Out[9]:
	key1	value_left	key2	value_right
0	foo	1	foo	5
1	foo	1	foo	8
2	foo	5	foo	5
3	foo	5	foo	8
4	bar	2	bar	6
5	baz	3	baz	7
Joining Dataframes
In [13]:
onedf = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                   'A': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
In [14]:
otherdf = pd.DataFrame({'key': ['K0', 'K1', 'K2'],
                      'B': ['B0', 'B1', 'B2']})
In [15]:
#join above two dataframes using indices
onedf.join(otherdf, lsuffix='_caller', rsuffix='_other')
Out[15]:
	key_caller	A	key_other	B
0	K0	A0	K0	B0
1	K1	A1	K1	B1
2	K2	A2	K2	B2
3	K3	A3	NaN	NaN
4	K4	A4	NaN	NaN
5	K5	A5	NaN	NaN
In [ ]:
#If we want to join using the key columns, we need to set key to be the index in both dataframes 
#onedf and otherdf. 
#The joined DataFrame will have key as its index.
In [16]:
onedf.set_index('key').join(otherdf.set_index('key'))
Out[16]:
	A	B
key		
K0	A0	B0
K1	A1	B1
K2	A2	B2
K3	A3	NaN
K4	A4	NaN
K5	A5	NaN
In [ ]:
#Another option to join using the key columns is to use the on parameter. 
#DataFrame.join always uses other’s index but we can use any column in df. 
#This method preserves the original DataFrame’s index in the result.
In [17]:
onedf.join(otherdf.set_index('key'), on='key')
Out[17]:
	key	A	B
0	K0	A0	B0
1	K1	A1	B1
2	K2	A2	B2
3	K3	A3	NaN
4	K4	A4	NaN
5	K5	A5	NaN
Groupby
In [18]:
df = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
                              'Parrot', 'Parrot'],
                   'Max Speed': [380., 370., 24., 26.]})
In [19]:
df
Out[19]:
	Animal	Max Speed
0	Falcon	380.0
1	Falcon	370.0
2	Parrot	24.0
3	Parrot	26.0
In [20]:
df.groupby(['Animal']).mean()
Out[20]:
	Max Speed
Animal	
Falcon	375.0
Parrot	25.0
In [ ]:
#Hierarchical Indexes
#We can groupby different levels of a hierarchical index using the level parameter:
In [24]:
arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],
          ['Captive', 'Wild', 'Captive', 'Wild']]
index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))
In [27]:
df = pd.DataFrame({'Max Speed': [390., 350., 30., 20.]},index=index)
In [28]:
df
Out[28]:
		Max Speed
Animal	Type	
Falcon	Captive	390.0
	Wild	350.0
Parrot	Captive	30.0
	Wild	20.0
In [29]:
df.groupby(level="Type").mean()
Out[29]:
	Max Speed
Type	
Captive	210.0
Wild	185.0
In [ ]:
 








Program 8
1	Problem Statement 
•	Write a python program to apply Student T test to perform hypothesis testing
2	Student Learning Outcomes
After successful completion of this lab, the student shall be able to
•	Formulate a hypothesis
•	Apply python functions for T-sets to test a hypothesis.
3	Design of the Program 
3.1	Description
What is t-test?

The t test (also called Student’s T Test) compares two averages (means) and tells you if they are different from each other. The t-test also tells you how significant the differences are; In other words it lets you know if those differences could have happened by chance.

A very simple example: Let’s say you have a cold and you try a naturalistic remedy. Your cold lasts a couple of days. The next time you have a cold, you buy an over-the-counter pharmaceutical and the cold lasts a week. You survey your friends and they all tell you that their colds were of a shorter duration (an average of 3 days) when they took the homeopathic remedy. What you really want to know is, are these results repeatable? A t-test can tell you by comparing the means of the two groups and letting you know the probability of those results happening by chance.

What is t-score?
The t score is a ratio between the difference between two groups and the difference within the groups. The larger the t score, the more difference there is between groups. The smaller the t score, the more similarity there is between groups. A t score of 3 means that the groups are three times as different from each other as they are within each other. When you run a t test, the bigger the t-value, the more likely it is that the results are repeatable.A large t-score tells you that the groups are different.A small t-score tells you that the groups are similar


What are T-Values and P-values?
Every t-value has a p-value to go with it. A p-value is the probability that the results from your sample data occurred by chance. P-values are from 0% to 100%. They are usually written as a decimal. For example, a p value of 5% is 0.05. Low p-values are good; They indicate your data did not occur by chance. For example, a p-value of .01 means there is only a 1% probability that the results from an experiment happened by chance. In most cases, a p-value of 0.05 (5%) is accepted to mean the data is valid.

Types of t-tests?
There are three main types of t-test:
1. An Independent Samples t-test compares the means for two groups.
2. A Paired sample t-test compares means from the same group at different times (say, one year apart).
3. A One sample t-test tests the mean of a single group against a known mean.

How to perform a 2 sample t-test?
Lets us say we have to test whether the height of men in the population is different from height of women in general. So we take a sample from the population and use the t-test to see if the result is significant.

3.2	Algorithm 
1.	Determine a null and alternate hypothesis.
In general, the null hypothesis will state that the two populations being tested have no statistically significant difference. The alternate hypothesis will state that there is one present. In this example we can say that:
2. Collect sample data
3. Determine a confidence interval and degrees of freedom
4. Calculate the t-statistic
5. Calculate the critical t-value from the t distribution
6. Compare the critical t-values with the calculated t statistic
3.3	Python Code 
## Import the packages
import numpy as np
from scipy import stats


## Define 2 random distributions
#Sample Size is 10
N = 10
#Gaussian distributed data with mean = 2 and var = 1
a = np.random.randn(N) + 2
#Gaussian distributed data with with mean = 0 and var = 1
b = np.random.randn(N)

## Calculate the Standard Deviation from variance
#For unbiased max likelihood estimate we have to divide the var by N-1,and therefore the parameter ddof = 1

var_a = a.var(ddof=1)
var_b = b.var(ddof=1)

#std deviation is square root of variance
s = np.sqrt((var_a + var_b)/2)
print('Standard Deviation',s)

## Calculate the t-statistics
print('T statistic and p value calculated manually')
t = (a.mean() - b.mean())/(s*np.sqrt(2/N))
print('t-statistic',t)

## Compare with the critical t-value calculated by scipy function stats.t.cdf() 
#Degrees of freedom
deg_f = 2*N - 2
#Calculate p-value  after comparison with the t 
p = 1 - stats.t.cdf(t,df=deg_f)
print('p-value',p)
print("t statistic= " + str(t))
print("p value= " + str(2*p))

### You can see that after comparing the t statistic with the critical t value
#(computed internally) we get a good p value of 0.0005 and thus we reject the null hypothesis 
#and thus it proves that the mean of the two distributions are different and statistically significant.
## Cross Checking with the internal scipy function

print('T statistic and p value from scipy stats library function for t-test')
t2, p2 = stats.ttest_ind(a,b)
print("t = " + str(t2))
print("p = " + str(p2))
print('Therefore we see that manually calculated and internally calculated values of t-statistic and p-value match')

Expected Output

Standard Deviation 1.1216971672485267
T statistic and p value calculated manually
t-statistic 5.488986814830812
p-value 1.6313585847416157e-05
t statistic= 5.488986814830812
p value= 3.262717169483231e-05
T statistic and p value from scipy stats library function for t-test
t = 5.488986814830812
p = 3.2627171694848136e-05
Therefore we see that manually calculated and internally calculated values of t-statistic and p-value match













Program 9
1	Problem Statement 
Write a python program to apply simple linear regression on data
2	Student Learning Outcomes
After successful completion of this lab, the student shall be able to
•	Import the required module from sklearn
•	Load data in required format
•	Apply the linear regression functions for fitting the model, predicting the value of response variable and printing the scores of the model
3	Design of the Program
3.1	Program Description
Regression Analysis is a statistical technique that searches for relationships among variables.
For example, you can observe several employees of some company and try to understand how their salaries depend on the features, such as experience, level of education, role, city they work in, and so on.
This is a regression problem where data related to each employee represent one observation. The presumption is that the experience, education, role, and city are the independent features, while the salary depends on them.
Similarly, you can try to establish a mathematical dependence of the prices of houses on their areas, numbers of bedrooms, distances to the city center, and so on.
Generally, in regression analysis, we try to establish a mathematical relation among two or more features in a dataset.  The dependent features are called the dependent variables, outputs, or responses.The independent features are called the independent variables, inputs, or predictors. Regression problems usually have one continuous and unbounded dependent variable. The inputs, however, can be continuous, discrete, or even categorical data such as gender, nationality, brand, and so on.
It is a common practice to denote the outputs with 𝑦 and inputs with 𝑥. If there are two or more independent variables, they can be represented as the vector 𝐱 = (𝑥₁, …, 𝑥ᵣ), where 𝑟 is the number of inputs. When implementing linear regression of some dependent variable 𝑦 on the set of independent variables 𝐱 = (𝑥₁, …, 𝑥ᵣ), where 𝑟 is the number of predictors, you assume a linear relationship between 𝑦 and 𝐱: 𝑦 = 𝛽₀ + 𝛽₁𝑥₁ + ⋯ + 𝛽ᵣ𝑥ᵣ + 𝜀. This equation is the regression equation. 𝛽₀, 𝛽₁, …, 𝛽ᵣ are the regression coefficients, and 𝜀 is the random error.  Linear regression calculates the estimators of the regression coefficients or simply the predicted weights, denoted with 𝑏₀, 𝑏₁, …, 𝑏ᵣ. They define the estimated regression function 𝑓(𝐱) = 𝑏₀ + 𝑏₁𝑥₁ + ⋯ + 𝑏ᵣ𝑥ᵣ. This function should capture the dependencies between the inputs and output sufficiently well. The estimated or predicted response, 𝑓(𝐱ᵢ), for each observation 𝑖 = 1, …, 𝑛, should be as close as possible to the corresponding actual response 𝑦ᵢ. The differences 𝑦ᵢ - 𝑓(𝐱ᵢ) for all observations 𝑖 = 1, …, 𝑛, are called the residuals. Regression is about determining the best predicted weights, that is the weights corresponding to the smallest residuals. To get the best weights, you usually minimize the sum of squared residuals (SSR) for all observations 𝑖 = 1, …, 𝑛: SSR = Σᵢ(𝑦ᵢ - 𝑓(𝐱ᵢ))². This approach is called the method of ordinary least squares.
Simple Linear Regression
Simple or single-variate linear regression is the simplest case of linear regression with a single independent variable, 𝐱 = 𝑥.
 

3.2	Algorithm
There are five basic steps when you’re implementing linear regression:
1.	Import the packages and classes you need.
2.	Provide data to work with and eventually do appropriate transformations.
3.	Create a regression model and fit it with existing data.
4.	Check the results of model fitting to know whether the model is satisfactory.
5.	Apply the model for predictions
3.3	Python Code 

# import numpy for creating arrays
# import sklearn's LinearRegression class 
import numpy as np
from sklearn.linear_model import LinearRegression
In [2]:
# create two data arrays, x and y using numpy arrays
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])
In [3]:
# create an object of the model of LinearRegression()
model = LinearRegression()
In [4]:
# fitting the model to x and y
model.fit(x, y)
Expected Output:
Out[4]:
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
In [5]:
# Printing the performance of model in terms of r squared value
r_sq = model.score(x, y)
Expected Output:
print('coefficient of determination:', r_sq)
coefficient of determination: 0.715875613747954

In [6]:
# Printing intercept and slope calculated by the system
print('intercept:', model.intercept_)
print('slope:', model.coef_)
Expected Output:

intercept: 5.633333333333329
slope: [0.54]

In [7]:
# Response predicted using internal predict() function
y_pred = model.predict(x)
print('predicted response using builtin predict() function:', y_pred, sep='\n')
Expected Output:

predicted response:
[ 8.33333333 13.73333333 19.13333333 24.53333333 29.93333333 35.33333333]
In [8]:
#Response predicted using manual calculation on intercept and coefficient
y_pred = model.intercept_ + model.coef_ * x
print('predicted response using manual calculation on intercept and coefficients:', y_pred, sep='\n')

Expected Output:

predicted response:
[[ 8.33333333]
 [13.73333333]
 [19.13333333]
 [24.53333333]
 [29.93333333]
 [35.33333333]]
In [9]:
# Create a new X array
x_new = np.arange(5).reshape((-1, 1))
print(x_new)
Expected Output:

[[0]
 [1]
 [2]
 [3]
 [4]]
In [10]:
# Predict values of y for this new X
y_new = model.predict(x_new)
print(y_new)
Expected Output:

[5.63333333 6.17333333 6.71333333 7.25333333 7.79333333]

























Program 10
1	Problem Statement 
Write a python program to apply multiple linear regression on data
2	Student Learning Outcomes
After successful completion of this lab, the student shall be able to
•	Import the required module from sklearn
•	Load data in required format
•	Apply the multiple linear regression functions for fitting the model on required features
•	predicting the value of response variable and printing the scores of the model
3	Design of the Program 
3.1	Description
Multiple regression is like linear regression, but with more than one independent value, meaning that we try to predict a value based on two or more variables.Multiple linear regression (MLR), also known simply as multiple regression, is a statistical technique that uses several explanatory variables to predict the outcome of a response variable. The goal of multiple linear regression (MLR) is to model the linear relationship between the explanatory (independent) variables and response (dependent) variable.In essence, multiple regression is the extension of ordinary least-squares (OLS) regression that involves more than one explanatory variable.

3.2	Algorithm 
1.	Import packages and classes, and provide data
2.	Create a model and fit it
3.	Predit the values of response variables
4.	Print score of the model in terms of squared error
3.3	Python Code 
import pandas
In [3]:
df = pandas.read_csv("cars.csv")
In [4]:
df
Expected Output:
Out[4]:
	Car	Model	Volume	Weight	CO2
0	Toyoty	Aygo	1000	790	99
1	Mitsubishi	Space Star	1200	1160	95
2	Skoda	Citigo	1000	929	95
3	Fiat	500	900	865	90
4	Mini	Cooper	1500	1140	105
5	VW	Up!	1000	929	105
6	Skoda	Fabia	1400	1109	90
7	Mercedes	A-Class	1500	1365	92
8	Ford	Fiesta	1500	1112	98
9	Audi	A1	1600	1150	99
10	Hyundai	I20	1100	980	99
11	Suzuki	Swift	1300	990	101
12	Ford	Fiesta	1000	1112	99
13	Honda	Civic	1600	1252	94
14	Hundai	I30	1600	1326	97
15	Opel	Astra	1600	1330	97
16	BMW	1	1600	1365	99
17	Mazda	3	2200	1280	104
18	Skoda	Rapid	1600	1119	104
19	Ford	Focus	2000	1328	105
20	Ford	Mondeo	1600	1584	94
21	Opel	Insignia	2000	1428	99
22	Mercedes	C-Class	2100	1365	99
23	Skoda	Octavia	1600	1415	99
24	Volvo	S60	2000	1415	99
25	Mercedes	CLA	1500	1465	102
26	Audi	A4	2000	1490	104
27	Audi	A6	2000	1725	114
28	Volvo	V70	1600	1523	109
29	BMW	5	2000	1705	114
30	Mercedes	E-Class	2100	1605	115
31	Volvo	XC70	2000	1746	117
32	Ford	B-Max	1600	1235	104
33	BMW	216	1600	1390	108
34	Opel	Zafira	1600	1405	109
35	Mercedes	SLK	2500	1395	120
In [6]:
X = df[['Weight', 'Volume']]
y = df['CO2']
In [7]:
from sklearn import linear_model
In [8]:
regr = linear_model.LinearRegression()
regr.fit(X, y)
Expected Output:

Out[8]:
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
In [9]:
#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300ccm:
predictedCO2 = regr.predict([[2300, 1300]])
In [11]:
print(predictedCO2)
Expected Output:
[107.2087328]

In [10]:
print(regr.coef_)
Expected Output:
[0.00755095 0.00780526]
The result array represents the coefficient values of weight and volume.
Weight: 0.00755095 Volume: 0.00780526
These values tell us that if the weight increase by 1kg, the CO2 emission increases by 0.00755095g.
And if the engine size (Volume) increases by 1 ccm, the CO2 emission increases by 0.00780526 g.
I think that is a fair guess, but let test it!
We have already predicted that if a car with a 1300ccm engine weighs 2300kg, the CO2 emission will be approximately 107g. What if we increase the weight with 1000kg?
In [12]:
#Predicted CO2 for higher weight
predictedCO2 = regr.predict([[3300, 1300]])
print(predictedCO2)
Expected Output:
[114.75968007]
We have predicted that a car with 1.3 liter engine, and a weight of 3300 kg, will release approximately 115 grams of CO2 for every kilometer it drives. Which shows that the coefficient of 0.00755095 is correct:
107.2087328 + (1000 * 0.00755095) = 114.75968
 




Program 11
1	Problem Statement 
Ability to import specific columns and convert data types is often a powerful transformation that helps during data analysis. Demonstrate following on Boston Dataset - Import every 50th row of Boston Housing dataset as a dataframe, then Import the Boston housing dataset, but while importing change the 'medv' (median house value) column so that values < 25 becomes ‘Low’ and > 25 becomes ‘High’
2	Student Learning Outcomes
After successful completion of this lab, the student shall be able to
•	Import numpy, pandas and other required modules
•	Use correct function and logic for loading required data
3	Design of the Program 
3.1	Description
We can use chunksize option while reading the data to load specific rows. Converters option with a lambda function can  be used for converting from numeric to categorical value.
3.2	Algorithm 
1.	Import required modules
2.	Use read_csv() to load Boston housing data
3.	Use chunksize option to load required row numbers
4.	Use converters option to convert the data format
3.3	Python Code 
Program 11
In [2]:
#First let us import the full datafile
import pandas as pd
df1 = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')
In [17]:
df1
Out[17]:
	crim	zn	indus	chas	nox	rm	age	dis	rad	tax	ptratio	b	lstat	medv
0	0.00632	18.0	2.31	0	0.538	6.575	65.2	4.0900	1	296	15.3	396.90	4.98	24.0
1	0.02731	0.0	7.07	0	0.469	6.421	78.9	4.9671	2	242	17.8	396.90	9.14	21.6
2	0.02729	0.0	7.07	0	0.469	7.185	61.1	4.9671	2	242	17.8	392.83	4.03	34.7
3	0.03237	0.0	2.18	0	0.458	6.998	45.8	6.0622	3	222	18.7	394.63	2.94	33.4
4	0.06905	0.0	2.18	0	0.458	7.147	54.2	6.0622	3	222	18.7	396.90	5.33	36.2
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
501	0.06263	0.0	11.93	0	0.573	6.593	69.1	2.4786	1	273	21.0	391.99	9.67	22.4
502	0.04527	0.0	11.93	0	0.573	6.120	76.7	2.2875	1	273	21.0	396.90	9.08	20.6
503	0.06076	0.0	11.93	0	0.573	6.976	91.0	2.1675	1	273	21.0	396.90	5.64	23.9
504	0.10959	0.0	11.93	0	0.573	6.794	89.3	2.3889	1	273	21.0	393.45	6.48	22.0
505	0.04741	0.0	11.93	0	0.573	6.030	80.8	2.5050	1	273	21.0	396.90	7.88	11.9
506 rows × 14 columns
In [18]:
# let us now import every 50th row of BostonHousing dataset as a dataframe
# Solution 1: Use chunks and for-loop
import pandas as pd
dfchunks = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', chunksize=50)
df2 = pd.DataFrame()
for chunk in dfchunks:
    df2 = df2.append(chunk.iloc[0,:])    
In [19]:
df2
Out[19]:
	age	b	chas	crim	dis	indus	lstat	medv	nox	ptratio	rad	rm	tax	zn
0	65.2	396.90	0.0	0.00632	4.0900	2.31	4.98	24.0	0.538	15.3	1.0	6.575	296.0	18.0
50	45.7	395.56	0.0	0.08873	6.8147	5.64	13.45	19.7	0.439	16.8	4.0	5.963	243.0	21.0
100	79.9	394.76	0.0	0.14866	2.7778	8.56	9.42	27.5	0.520	20.9	5.0	6.727	384.0	0.0
150	97.3	372.80	0.0	1.65660	1.6180	19.58	14.10	21.5	0.871	14.7	5.0	6.122	403.0	0.0
200	13.9	384.30	0.0	0.01778	7.6534	1.47	4.45	32.9	0.403	17.0	3.0	7.135	402.0	95.0
250	13.0	396.28	0.0	0.14030	7.3967	5.86	5.90	24.4	0.431	19.1	7.0	6.487	330.0	22.0
300	47.4	390.86	0.0	0.04417	7.8278	2.24	6.07	24.8	0.400	14.8	5.0	6.871	358.0	70.0
350	44.4	396.90	0.0	0.06211	8.7921	1.25	5.98	22.9	0.429	19.7	1.0	6.490	335.0	40.0
400	100.0	396.90	0.0	25.04610	1.5888	18.10	26.77	5.6	0.693	20.2	24.0	5.987	666.0	0.0
450	92.6	0.32	0.0	6.71772	2.3236	18.10	17.44	13.4	0.713	20.2	24.0	6.749	666.0	0.0
500	79.7	396.90	0.0	0.22438	2.4982	9.69	14.33	16.8	0.585	19.2	6.0	6.027	391.0	0.0
In [20]:
#part 2 of solution- convert medv from numerical to categorical format
import pandas as pd
df3 = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', 
                 converters={'medv': lambda x: 'High' if float(x) > 25 else 'Low'})
df3
Out[20]:
	crim	zn	indus	chas	nox	rm	age	dis	rad	tax	ptratio	b	lstat	medv
0	0.00632	18.0	2.31	0	0.538	6.575	65.2	4.0900	1	296	15.3	396.90	4.98	Low
1	0.02731	0.0	7.07	0	0.469	6.421	78.9	4.9671	2	242	17.8	396.90	9.14	Low
2	0.02729	0.0	7.07	0	0.469	7.185	61.1	4.9671	2	242	17.8	392.83	4.03	High
3	0.03237	0.0	2.18	0	0.458	6.998	45.8	6.0622	3	222	18.7	394.63	2.94	High
4	0.06905	0.0	2.18	0	0.458	7.147	54.2	6.0622	3	222	18.7	396.90	5.33	High
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
501	0.06263	0.0	11.93	0	0.573	6.593	69.1	2.4786	1	273	21.0	391.99	9.67	Low
502	0.04527	0.0	11.93	0	0.573	6.120	76.7	2.2875	1	273	21.0	396.90	9.08	Low
503	0.06076	0.0	11.93	0	0.573	6.976	91.0	2.1675	1	273	21.0	396.90	5.64	Low
504	0.10959	0.0	11.93	0	0.573	6.794	89.3	2.3889	1	273	21.0	393.45	6.48	Low
505	0.04741	0.0	11.93	0	0.573	6.030	80.8	2.5050	1	273	21.0	396.90	7.88	Low
506 rows × 14 columns
In [ ]:
 






Program 12
1	Problem Statement 
Data analysis aims to find interesting insights from the data. Using the Boston Housing Dataset, apply the learnt concepts of data summarization, visualization, correlation and regression to derive your own interesting insights from the data. The results of this program may vary from student to student.
2	Student Learning Outcomes
After successful completion of this lab, the student shall be able to
•	Apply the learnt concepts to draw interesting insights from dataset
3	Design of the Program 
3.1	Description
The following describes the dataset columns:
•	CRIM - per capita crime rate by town
•	ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
•	INDUS - proportion of non-retail business acres per town.
•	CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
•	NOX - nitric oxides concentration (parts per 10 million)
•	RM - average number of rooms per dwelling
•	AGE - proportion of owner-occupied units built prior to 1940
•	DIS - weighted distances to five Boston employment centres
•	RAD - index of accessibility to radial highways
•	TAX - full-value property-tax rate per $10,000
•	PTRATIO - pupil-teacher ratio by town
•	B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
•	LSTAT - % lower status of the population
•	MEDV - Median value of owner-occupied homes in $1000's
We can apply describe() to see summary of entire dataset. Apply corr() to find correlation among columns of interest. Use visualization to find interesting patterns. 
3.2	Algorithm 
1.	Import the required modules
2.	Import dataset
3.	Apply describe()
4.	Apply corr() to interesting columns
5.	Apply visualization to find interesting patterns and outliers
6.	Apply regression to find MSE
7.	Transform Data and see the effect on MSE
8.	Write a report on your insight from data
3.3	Python Code 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv('BostonHousing.csv')
data
Out[1]:
	crim	zn	indus	chas	nox	rm	age	dis	rad	tax	ptratio	b	lstat	medv
0	0.00632	18.0	2.31	0	0.538	6.575	65.2	4.0900	1	296	15.3	396.90	4.98	24.0
1	0.02731	0.0	7.07	0	0.469	6.421	78.9	4.9671	2	242	17.8	396.90	9.14	21.6
2	0.02729	0.0	7.07	0	0.469	7.185	61.1	4.9671	2	242	17.8	392.83	4.03	34.7
3	0.03237	0.0	2.18	0	0.458	6.998	45.8	6.0622	3	222	18.7	394.63	2.94	33.4
4	0.06905	0.0	2.18	0	0.458	7.147	54.2	6.0622	3	222	18.7	396.90	5.33	36.2
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
501	0.06263	0.0	11.93	0	0.573	6.593	69.1	2.4786	1	273	21.0	391.99	9.67	22.4
502	0.04527	0.0	11.93	0	0.573	6.120	76.7	2.2875	1	273	21.0	396.90	9.08	20.6
503	0.06076	0.0	11.93	0	0.573	6.976	91.0	2.1675	1	273	21.0	396.90	5.64	23.9
504	0.10959	0.0	11.93	0	0.573	6.794	89.3	2.3889	1	273	21.0	393.45	6.48	22.0
505	0.04741	0.0	11.93	0	0.573	6.030	80.8	2.5050	1	273	21.0	396.90	7.88	11.9
506 rows × 14 columns
In [2]:
data.shape
Out[2]:
(506, 14)
In [3]:
data.describe()
Out[3]:
	crim	zn	indus	chas	nox	rm	age	dis	rad	tax	ptratio	b	lstat	medv
count	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000
mean	3.613524	11.363636	11.136779	0.069170	0.554695	6.284634	68.574901	3.795043	9.549407	408.237154	18.455534	356.674032	12.653063	22.532806
std	8.601545	23.322453	6.860353	0.253994	0.115878	0.702617	28.148861	2.105710	8.707259	168.537116	2.164946	91.294864	7.141062	9.197104
min	0.006320	0.000000	0.460000	0.000000	0.385000	3.561000	2.900000	1.129600	1.000000	187.000000	12.600000	0.320000	1.730000	5.000000
25%	0.082045	0.000000	5.190000	0.000000	0.449000	5.885500	45.025000	2.100175	4.000000	279.000000	17.400000	375.377500	6.950000	17.025000
50%	0.256510	0.000000	9.690000	0.000000	0.538000	6.208500	77.500000	3.207450	5.000000	330.000000	19.050000	391.440000	11.360000	21.200000
75%	3.677082	12.500000	18.100000	0.000000	0.624000	6.623500	94.075000	5.188425	24.000000	666.000000	20.200000	396.225000	16.955000	25.000000
max	88.976200	100.000000	27.740000	1.000000	0.871000	8.780000	100.000000	12.126500	24.000000	711.000000	22.000000	396.900000	37.970000	50.000000
In [15]:
# Do boxplot and find outliers
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in data.items():
    sns.boxplot(y=k, data=data, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

In [17]:
#Calculate the percentage of outliers in each column of dataset
for k, v in data.items():
        q1 = v.quantile(0.25)
        q3 = v.quantile(0.75)
        irq = q3 - q1
        v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
        perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
        print("Column %s outliers = %.2f%%" % (k, perc))
Column crim outliers = 13.04%
Column zn outliers = 13.44%
Column indus outliers = 0.00%
Column chas outliers = 100.00%
Column nox outliers = 0.00%
Column rm outliers = 5.93%
Column age outliers = 0.00%
Column dis outliers = 0.99%
Column rad outliers = 0.00%
Column tax outliers = 0.00%
Column ptratio outliers = 2.96%
Column b outliers = 15.22%
Column lstat outliers = 1.38%
Column medv outliers = 7.91%
In [13]:
data.corr()
Out[13]:
	crim	zn	indus	chas	nox	rm	age	dis	rad	tax	ptratio	b	lstat	medv
crim	1.000000	-0.200469	0.406583	-0.055892	0.420972	-0.219247	0.352734	-0.379670	0.625505	0.582764	0.289946	-0.385064	0.455621	-0.388305
zn	-0.200469	1.000000	-0.533828	-0.042697	-0.516604	0.311991	-0.569537	0.664408	-0.311948	-0.314563	-0.391679	0.175520	-0.412995	0.360445
indus	0.406583	-0.533828	1.000000	0.062938	0.763651	-0.391676	0.644779	-0.708027	0.595129	0.720760	0.383248	-0.356977	0.603800	-0.483725
chas	-0.055892	-0.042697	0.062938	1.000000	0.091203	0.091251	0.086518	-0.099176	-0.007368	-0.035587	-0.121515	0.048788	-0.053929	0.175260
nox	0.420972	-0.516604	0.763651	0.091203	1.000000	-0.302188	0.731470	-0.769230	0.611441	0.668023	0.188933	-0.380051	0.590879	-0.427321
rm	-0.219247	0.311991	-0.391676	0.091251	-0.302188	1.000000	-0.240265	0.205246	-0.209847	-0.292048	-0.355501	0.128069	-0.613808	0.695360
age	0.352734	-0.569537	0.644779	0.086518	0.731470	-0.240265	1.000000	-0.747881	0.456022	0.506456	0.261515	-0.273534	0.602339	-0.376955
dis	-0.379670	0.664408	-0.708027	-0.099176	-0.769230	0.205246	-0.747881	1.000000	-0.494588	-0.534432	-0.232471	0.291512	-0.496996	0.249929
rad	0.625505	-0.311948	0.595129	-0.007368	0.611441	-0.209847	0.456022	-0.494588	1.000000	0.910228	0.464741	-0.444413	0.488676	-0.381626
tax	0.582764	-0.314563	0.720760	-0.035587	0.668023	-0.292048	0.506456	-0.534432	0.910228	1.000000	0.460853	-0.441808	0.543993	-0.468536
ptratio	0.289946	-0.391679	0.383248	-0.121515	0.188933	-0.355501	0.261515	-0.232471	0.464741	0.460853	1.000000	-0.177383	0.374044	-0.507787
b	-0.385064	0.175520	-0.356977	0.048788	-0.380051	0.128069	-0.273534	0.291512	-0.444413	-0.441808	-0.177383	1.000000	-0.366087	0.333461
lstat	0.455621	-0.412995	0.603800	-0.053929	0.590879	-0.613808	0.602339	-0.496996	0.488676	0.543993	0.374044	-0.366087	1.000000	-0.737663
medv	-0.388305	0.360445	-0.483725	0.175260	-0.427321	0.695360	-0.376955	0.249929	-0.381626	-0.468536	-0.507787	0.333461	-0.737663	1.000000
In [14]:
#Correlation of each feature with the target variable
data[data.columns[1:]].corr()['medv'][:]
Out[14]:
zn         0.360445
indus     -0.483725
chas       0.175260
nox       -0.427321
rm         0.695360
age       -0.376955
dis        0.249929
rad       -0.381626
tax       -0.468536
ptratio   -0.507787
b          0.333461
lstat     -0.737663
medv       1.000000
Name: medv, dtype: float64
In [12]:
#Produce correlation matrix
sns.heatmap(data.corr().abs(),  annot=True)
Out[12]:
<matplotlib.axes._subplots.AxesSubplot at 0x1dbd6229648>

In [23]:
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
import numpy as np
# We plot columns with highest correlation  to medvLet's scale the columns before plotting them against MEDV
cols=['indus','nox','rm','tax','ptratio','lstat']
x = data.loc[:,cols]
y = data['medv']

reg = linear_model.LinearRegression().fit(x, y)
reg.score(x, y)
print(reg.coef_)
print(reg.intercept_)
[ 8.71873392e-02 -3.40311735e+00  4.65592779e+00 -2.90110504e-03
 -9.13819473e-01 -5.45934588e-01]
19.145818460578578
In [24]:
x
Out[24]:
	indus	nox	rm	tax	ptratio	lstat
0	2.31	0.538	6.575	296	15.3	4.98
1	7.07	0.469	6.421	242	17.8	9.14
2	7.07	0.469	7.185	242	17.8	4.03
3	2.18	0.458	6.998	222	18.7	2.94
4	2.18	0.458	7.147	222	18.7	5.33
...	...	...	...	...	...	...
501	11.93	0.573	6.593	273	21.0	9.67
502	11.93	0.573	6.120	273	21.0	9.08
503	11.93	0.573	6.976	273	21.0	5.64
504	11.93	0.573	6.794	273	21.0	6.48
505	11.93	0.573	6.030	273	21.0	7.88
506 rows × 6 columns
In [29]:
#Apply preprocessing on x before fitting the model
x_scaled = pd.DataFrame(data=min_max_scaler.fit_transform(x))
reg = linear_model.LinearRegression().fit(x_scaled, y)
reg.score(x_scaled, y)
print(reg.coef_)
print(reg.intercept_)
[  2.37847061  -1.65391503  24.29928711  -1.52017904  -8.58990305
 -19.78466946]
21.454384458131905
In [34]:
kf = KFold(n_splits=10)
scores = cross_val_score(reg, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
print("Mean Squared Error: Mean MSE: %0.2f Std Error: (+/- %0.2f)" % (scores.mean(), scores.std()))
Mean Squared Error: Mean MSE: -36.56 Std Error: (+/- 46.35)
In [38]:
#Data Transformation - Removing Skew from data
y =  np.log1p(y)
for col in x.columns:
    if np.abs(x[col].skew()) > 0.3:
        x[col] = np.log1p(x[col])
In [39]:
#Apply regression again and see the effect on MSE
x_scaled = pd.DataFrame(data=min_max_scaler.fit_transform(x))
reg = linear_model.LinearRegression().fit(x_scaled, y)
kf = KFold(n_splits=10)
scores = cross_val_score(reg, x_scaled, y, cv=kf, scoring='neg_mean_squared_error')
print("Mean Squared Error: Mean MSE: %0.2f Std Error: (+/- %0.2f)" % (scores.mean(), scores.std()))
Mean Squared Error: Mean MSE: -0.05 Std Error: (+/- 0.06)
In [ ]:
 












Viva Voice Questions
1.	What is Jupyter Notebook?
2.	What is the difference between a bar chart and a histogram?
3.	What is a module in Python?
4.	How do you read a dataset from a CSV file into a dataframe?
5.	What do you mean by data analysis?
6.	How is data analysis different from Machine Learning?
7.	What are various data cleaning operations?
8.	What is the meaning of missing value?
9.	What are possible ways to impute missing values in a dataframe?
10.	What is a dataframe?
11.	 What does the term outlier mean?
12.	What are various types of correlation coefficients?
13.	What do you mean by a high positive correlation?
14.	What is hypothesis?
15.	What is a hypothesis test?
16.	What is data fishing?
17.	What is data wrangling?
18.	What is data dredging?
19.	Name essential python science libraries.
20.	What is a statistical model?
21.	What is the difference between simple linear and multiple linear regression?
22.	What is correlation?
23.	How do you gain meaningful insights from data?
24.	What do you mean by reshaping a data frame?
25.	What is automatic index alignment?
26.	What is a dataframe?
27.	What are various methods for accessing data from a dataframe?
28.	What is the difference between numpy and pandas?
29.	How do we compute transpose of a matrix?
30.	What are universal functions?
31.	How do we convert from datetime format to string format and vice versa?
32.	What do you mean by locale specific datetime formatting?
33.	What is time series data how do you obtain it?
34.	What is the purpose of GroupBy function?
35.	How do we merge two dataframes?
36.	What is broadcasting in Numpy?
37.	What is the difference between a Series object and a Dataframe object?
38.	Which function is used to change the index of a data frame?
39.	What is the difference between mutable and immutable data types in Python?
40.	What are various built in data structures in Python?
41.	What is the difference between Data Mining and Data Analysis?
42.	What is the difference between list and tuples in Python?
43.	What are the key features of Python?
44.	What type of language is python?
45.	How is Python an interpreted language?
46.	What is pep 8?
47.	How is memory managed in Python?
48.	What is name space in Python?
49.	 What is PYTHON PATH?
50.	 What are python modules?
51.	 What are local variables and global variables in Python?
******
