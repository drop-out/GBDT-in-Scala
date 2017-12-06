# GBDT in Scala
As one would expect, this is a `Scala` implementation of GBDT. 

I have implemented GBDT in `Python`, see [Machine-Learning-From-Scratch](https://github.com/drop-out/Machine-Learning-From-Scratch). And now I rewrite it to `Scala`. The structure of the code is almost the same as the `Python` implementation.

This implementation supports most of the core features of `xgboost`. Briefly, it supports:

- **Built-in loss**: Mean squared loss for regression task and log loss for classfication task.
- **Customized loss**: Other loss are also supported. User should provide the activation function, the gradient, and the hessian.
- **Hessian information**: It uses Newton Method for boosting, thus makes full use of the second-order derivative information. 
- **Regularization**: lambda and gamma, as in `xgboost`.
- **Multi-processing**: It uses `Scala` `parallel collection` for multi-processing

To keep the code neat, some features of `xgboost` are not implemented. For example, it does not handle missing value, and randomness is not supported.

## Dependence

The algorithm is implemented in `Scala 2.12`. The algorithm is build from scratch: It depends on `Breeze` for matrix calculation, and there is no dependence on any other machine learning package.

- [Breeze](https://github.com/scalanlp/breeze)

The project is build with `Maven`. So you can find the information about the dependencies in `pom.xml`.

## Comparison with Python implementation

As for peformance, Scala implementation is faster than [Python implementation with Numpy](https://github.com/drop-out/Machine-Learning-From-Scratch/blob/master/gbdt.py), tested on my machine. But it could not compare to the [Python implementation with Numba](https://github.com/drop-out/Machine-Learning-From-Scratch/blob/master/gbdt_numba.py). I have to say that `Numba` is really fast.

In terms of implementation, two remarks are in order. First, parallelization could be easily implemented in `Scala` with `parallel collection`, which makes the implementation process quite lovely. Second, I have to say that `Breeze` is not as mature as `Numpy`. It takes a lot of time to make `Breeze` code work.

## Usage

Refer to `example.scala` for usage.

**Initialize model**
```scala
var model = new GBDT(loss="mse",max_depth=3,min_sample_split=10,lambda=1.0,gamma=0.0,learning_rate=0.1,n_estimators=100)
```
* `loss`: Loss function for gradient boosting. `'mse'`  is mean squared error for regression task and `'log'` is log loss for classification task. 
* `loss_object`: Pass an object that extends the `loss` trait to use customized loss. See [source code](https://github.com/drop-out/GBDT-in-Scala/blob/master/src/main/scala/Tree.scala) for details.
* `max_depth`: The maximum depth of a tree.
* `min_sample_split`: The minimum number of samples required to further split a node.
* `lambda`: The regularization coefficient for leaf score, also known as lambda.
* `gamma`: The regularization coefficient for number of tree nodes, also know as gamma.
* `learning_rate`: The learning rate of gradient boosting.
* `n_estimators`: Number of trees.

**Train**
```scala
model.fit(train,target)
```
`train` should be a `DenseMatrix` of `Double`, and `target` should be a `DenseVector` of `Double`.

Returns `Unit`.

**Predict**
```scala
var prediction = model.predict(test)
```
`test` should be a `DenseMatrix` of `Double`.

Returns predictions as `DenseVector`.

**Customized loss**

Define a object that inheritates the `loss` trait (see [source code](https://github.com/drop-out/GBDT-in-Scala/blob/master/src/main/scala/Tree.scala)), which specifies the activation function, the gradients and the hessian. 

For example:

```scala
object log extends loss{
    def act(score:DenseVector[Double]): DenseVector[Double] ={
        1.0/(exp(-score)+1.0)
    }
    def g(score:DenseVector[Double],target:DenseVector[Double]):DenseVector[Double]={
        var pred = act(score)
        pred-target
    }
    def h(score:DenseVector[Double],target:DenseVector[Double]):DenseVector[Double]={
        var pred = act(score)
        pred*:*(1.0-pred)
    }
}
```

`g` is gradient and `h` is hessian.

And the object could be passed when initializing the model.

```scala
var model = new GBDT(loss_object=log,learning_rate=0.1,n_estimators=100)
```
