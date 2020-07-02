import cv2
import numpy as np
from collections import defaultdict

img = cv2.imread('original.jpg')
cv2.imshow("original image",img)
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray image',gray_img)
cv2.imwrite('gray_image.jpg',gray_img)
retval, img = cv2.threshold(gray_img,246,255,cv2.THRESH_BINARY)
cv2.imwrite("binary_img.jpg",img)
cv2.imshow("black and white image",img)
img_num = np.array(img)
print(img_num.shape)

#img_num = np.array([[255,255,255,255],[255,0,0,255],[255,0,0,255],[255,255,255,255]])
dim = img_num.shape
graph = np.zeros((dim[0]*dim[1]+2,dim[0]*dim[1]+2))
print(graph.shape)
for i in range(dim[0]):
    for j in range(dim[1]):
        if j+1<dim[1]:
            graph[dim[1]*i+j+1][dim[1]*i+j+2] = graph[dim[1]*i+j+2][dim[1]*i+j+1] = max(1 - abs(int(img_num[i][j])-int(img_num[i][j+1])),0.1)
        if i+1<dim[0]:
            graph[dim[1]*(i+1)+j+1][dim[1]*i+j+1] = graph[dim[1]*i+j+1][dim[1]*(i+1)+j+1] = max(1 - abs(int(img_num[i][j])-int(img_num[i+1][j])),0.1)
        if img_num[i][j] == 255 :
            graph[0][dim[1]*i+j+1] = 1000000
        else:
            graph[dim[1]*i+j+1][dim[0]*dim[1]+1] = 1000000

graph = graph.tolist()
# This class represents a directed graph using adjacency matrix representation 
class Graph: 
  
    def __init__(self,graph): 
        self.graph = graph # residual graph 
        self.org_graph = [i[:] for i in graph] 
        self. ROW = len(graph) 
        self.COL = len(graph[0])
        self.edges = []
    def BFS(self,s, t, parent): 
  
        # Mark all the vertices as not visited 
        visited =[False]*(self.ROW) 
  
        # Create a queue for BFS 
        queue=[] 
  
        # Mark the source node as visited and enqueue it 
        queue.append(s) 
        visited[s] = True
  
         # Standard BFS Loop x
        while queue: 
  
            #Dequeue a vertex from queue and print it 
            u = queue.pop(0) 
  
            # Get all adjacent vertices of the dequeued vertex u 
            # If a adjacent has not been visited, then mark it 
            # visited and enqueue it 
            for ind, val in enumerate(self.graph[u]): 
                if visited[ind] == False and val > 0 : 
                    queue.append(ind) 
                    visited[ind] = True
                    parent[ind] = u 
  
        # If we reached sink in BFS starting from source, then return 
        # true, else false 
        return True if visited[t] else False
  
  
    # Returns the min-cut of the given graph 
    def minCut(self, source, sink): 
  
        # This array is filled by BFS and to store path 
        parent = [-1]*(self.ROW) 
  
        max_flow = 0 # There is no flow initially 
  
        # Augment the flow while there is path from source to sink 
        while self.BFS(source, sink, parent) : 
  
            # Find minimum residual capacity of the edges along the 
            # path filled by BFS. Or we can say find the maximum flow 
            # through the path found. 
            path_flow = float("Inf") 
            s = sink 
            while(s !=  source): 
                path_flow = min (path_flow, self.graph[parent[s]][s]) 
                s = parent[s] 
  
            # Add path flow to overall flow 
            max_flow +=  path_flow 
  
            # update residual capacities of the edges and reverse edges 
            # along the path 
            v = sink 
            while(v !=  source): 
                u = parent[v] 
                self.graph[u][v] -= path_flow 
                self.graph[v][u] += path_flow 
                v = parent[v] 
  
        # print the edges which initially had weights 
        # but now have 0 weight
        for i in range(self.ROW): 
            for j in range(self.COL):
            
                if self.graph[i][j] == 0 and self.org_graph[i][j] > 0: 
                    self.edges.append((i,j))
  
  
# Create a graph given in the above diagram 

  
g = Graph(graph) 
  
source = 0
sink = dim[0]*dim[1]+1


g.minCut(source, sink)

#color_img =[ [[img_num[i][j],img_num[i][j],img_num[i][j]]for j in range(dim[1])]for i in range(dim[0])]


color_img = cv2.cvtColor(img_num, cv2.COLOR_GRAY2RGB)
for ele in g.edges:
    color_img[(ele[0]-1)//dim[1]][(ele[0]-1)%dim[1]] = [0,0,255]
cv2.imshow('ASDF',color_img)
cv2.imwrite("bounded_image.jpg",color_img)

