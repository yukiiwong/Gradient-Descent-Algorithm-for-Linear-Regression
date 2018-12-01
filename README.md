# Gradient-Descent-Algorithm-for-Linear-Regression
Seriously, it's the BGD(Batch-Gradient-Descent) algorithm.
## data.csv 
The data source used in the both programs.
## gradient_descent_numpy.py
At the beginning we need to import the data and determine the learning rate and the initial point.
```python
points = genfromtxt("data.csv", delimiter=",")
learning_rate 
initial_b 
initial_m 
num_iterations
```
Then, we need a cost function to calculate error.
```python
def compute_error_for_line_given_points(b, m, points)
```
Next, write a function to update the value of the parameter 𝜔. 
```python
def step_gradient(b_current, m_current, points, learningRate)
```
Finally, let it keep learning in a for loop.
```python
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations)
```
![plot](https://github.com/yukiiwong/Gradient-Descent-Algorithm-for-Linear-Regression/blob/master/gradient_descent_example.gif)
## gradient_descent_tensorflow.py
Use less code

A cost function
```python
loss = tf.reduce_mean(tf.square(tf.subtract(y, y_pred)))
```
Use GradientDescentOptimizer
```python
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    input, output = df[:, 0], df[:, 1]
    for epoch in range(numEpochs):
        _, sampleLoss, pred = session.run([optimiser, loss, y_pred], {x: input, y: output})

```
--------------------------------------------------------------------------------
## Conclusion
The disadvantage of the gradient descent method is that the convergence speed becomes slower at the minimum point and is extremely sensitive to the selection of the initial point. And what’s more, local optimum(局部最优) may occur.
