<H3> Name : TAMIZHSELVAN B </H3>
<H3>Register No. : 212223230225 </H3>
<H3> Experiment 1</H3>
<H1 ALIGN=CENTER> Implementation of Bayesian Networks</H1>

## Aim :
To create a bayesian Network for the given dataset in Python
    
## Algorithm:
Step 1:Import necessary libraries: pandas, networkx, matplotlib.pyplot, Bbn, Edge, EdgeType, BbnNode, Variable, EvidenceBuilder, InferenceController<br/>
Step 2:Set pandas options to display more columns<br/>
Step 3:Read in weather data from a CSV file using pandas<br/>
Step 4:Remove records where the target variable RainTomorrow has missing values<br/>
Step 5:Fill in missing values in other columns with the column mean<br/>
Step 6:Create bands for variables that will be used in the model (Humidity9amCat, Humidity3pmCat, and WindGustSpeedCat)<br/>
Step 7:Define a function to calculate probability distributions, which go into the Bayesian Belief Network (BBN)<br/>
Step 8:Create BbnNode objects for Humidity9amCat, Humidity3pmCat, WindGustSpeedCat, and RainTomorrow, using the probs() function to calculate their probabilities<br/>
Step 9:Create a Bbn object and add the BbnNode objects to it, along with edges between the nodes<br/>
Step 10:Convert the BBN to a join tree using the InferenceController<br/>
Step 11:Set node positions for the graph<br/>
Step 12:Set options for the graph appearance<br/>
Step 13:Generate the graph using networkx<br/>
Step 14:Update margins and display the graph using matplotlib.pyplot<br/>

## Program:
```py
!pip install pybbn
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd # for data manipulation
import networkx as nx # for drawing graphs
import matplotlib.pyplot as plt # for drawing graphs
# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
#Set Pandas options to display more columns
pd.options.display.max_columns=50


# Read in the weather data csv
df=pd.read_csv('/content/weatherAUS.csv', encoding='utf-8')


# Drop records where target RainTomorrow=NaN
df=df[pd.isnull(df['RainTomorrow'])==False]
# Drop the 'Date' column as it is not relevant for the model
df = df.drop(columns='Date')


# For other columns with missing values, fill them in with column mean for numeric columns only
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())


# Create bands for variables that we want to use in the model
df['WindGustSpeedCat']=df['WindGustSpeed'].apply(lambda x: '0.<=40'   if x<=40 else

                                         '1.40-50' if 40<x<=50 else '2.>50')
df['Humidity9amCat']=df['Humidity9am'].apply(lambda x: '1.>60' if x>60 else '0.<=60')
df['Humidity3pmCat']=df['Humidity3pm'].apply(lambda x: '1.>60' if x>60 else '0.<=60')



# Show a snaphsot of data
print(df)
def probs(data, child, parent1=None, parent2=None):
    if parent1 is None:
        # P(child)
        return (
            pd.crosstab(data[child], columns="count", normalize="columns")
            .sort_index()
            .to_numpy()
            .ravel()
            .tolist()
        )

    if parent2 is None:
        # P(child | parent1)
        table = pd.crosstab(data[parent1], data[child], normalize="index")
    else:
        # P(child | parent1, parent2)
        table = pd.crosstab([data[parent1], data[parent2]], data[child], normalize="index")

    return table.sort_index().to_numpy().ravel().tolist()
# Create nodes by using our earlier function to automatically calculate probabilities
H9am = BbnNode(Variable(0, 'H9am', ['<=60', '>60']), probs(df, child='Humidity9amCat'))
H3pm = BbnNode(Variable(1, 'H3pm', ['<=60', '>60']), probs(df, child='Humidity3pmCat', parent1='Humidity9amCat'))
W = BbnNode(Variable(2, 'W', ['<=40', '40-50', '>50']), probs(df, child='WindGustSpeedCat'))
RT = BbnNode(Variable(3, 'RT', ['No', 'Yes']), probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat'))


# Create Network
bbn = Bbn() \
    .add_node(H9am) \
    .add_node(H3pm) \
    .add_node(W) \
    .add_node(RT) \
    .add_edge(Edge(H9am, H3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(H3pm, RT, EdgeType.DIRECTED)) \
    .add_edge(Edge(W, RT, EdgeType.DIRECTED))

# Convert the BBN to a join tree
join_tree = InferenceController.apply(bbn)
# Set node positions
pos = {0: (-1, 2), 1: (-1, 0.5), 2: (1, 0.5), 3: (0, -1)}


# Set options for graph looks
options = {
    "font_size": 16,
    "node_size": 4000,
    "node_color": "white",
    "edgecolors": "black",
    "edge_color": "red",
    "linewidths": 5,
    "width": 5,}

# Generate graph
n, d = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=d, pos=pos, **options)


# Update margins and print the graph
ax = plt.gca()
ax.margins(0.10)
plt.axis("off")
plt.show()
print("CPTs: Humidity 9AM ->{}".format(probs(df, child='Humidity9amCat')))
print("CPTs: Humidity 3PM ->{}".format(probs(df, child='Humidity3pmCat', parent1='Humidity9amCat')))
print("CPTs: Wind Gust Speed ->{}".format(probs(df, child='WindGustSpeedCat')))
print("CPTs: Rain Tomorrow ->{}".format(probs(df, child='RainTomorrow', parent1='Humidity3pmCat', parent2='WindGustSpeedCat')))
rain_cpt = pd.crosstab(
    [df["Humidity3pmCat"], df["WindGustSpeedCat"]],
    df["RainTomorrow"],
    normalize="index"
)

print(rain_cpt.round(4))


```
## Output:

<img width="768" height="570" alt="image" src="https://github.com/user-attachments/assets/5499dee2-81cc-46df-8cbc-dea665df89db" />



<img width="755" height="437" alt="image" src="https://github.com/user-attachments/assets/96711566-2916-47b6-b364-47c28f24e7e3" />


<img width="902" height="641" alt="image" src="https://github.com/user-attachments/assets/7473d2d1-5d64-4ac7-ac4f-7a362c7715e1" />


<img width="1752" height="113" alt="image" src="https://github.com/user-attachments/assets/1beb6ce7-bec0-4a9e-b3c9-174cac8ce712" />


<img width="542" height="188" alt="image" src="https://github.com/user-attachments/assets/900bd08f-bb87-48b4-9386-a17589835c53" />

## Result:
   Thus a Bayesian Network is generated using Python

