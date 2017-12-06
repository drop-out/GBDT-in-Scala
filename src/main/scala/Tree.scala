import breeze.linalg._
import breeze.numerics._
import breeze.stats._


trait loss{
    /*
        The trait for loss function.
        Loss function or user defined loss function should extend this trait.
        Three things should be specified for a loss,
        namely activation function, gradient and hessian.
        act() is the activation function, which takes scores as input, and returns predictions.
        g() is the gradient, which takes true values and scores as input, and returns gradient.
        h() is the heassian, which takes true values and scores as input, and returns hessian.
    */
    def act(score:DenseVector[Double]): DenseVector[Double]
    def g(score:DenseVector[Double],target:DenseVector[Double]):DenseVector[Double]
    def h(score:DenseVector[Double],target:DenseVector[Double]):DenseVector[Double]
}


object mse extends loss{
    //Loss object for mse. As for mse, activation function is pred=score.
    def act(score:DenseVector[Double]): DenseVector[Double] ={
        score
    }
    def g(score:DenseVector[Double],target:DenseVector[Double]):DenseVector[Double]={
        score-target
    }
    def h(score:DenseVector[Double],target:DenseVector[Double]):DenseVector[Double]={
        DenseVector.fill(target.length){1}
    }
}

object log extends loss{
    //Loss class for log loss. As for log loss, activation function is logistic transformation.
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


class GBDT(var loss:String="mse",
           var loss_object:loss=null,
           var max_depth:Int=3,
           var min_sample_split:Int=10,
           var lambda:Double=1.0,
           var gamma:Double=0.0,
           var learning_rate:Double=0.1,
           var n_estimators:Int=100){
    /*
        Parameters:
        ----------
        loss: Loss function for gradient boosting.
            'mse' for regression task and 'log' for classfication task.
        loss_object:A class that extends the loss trait should be passed if you need customized loss.
        max_depth: The maximum depth of a tree.
        min_sample_split: The minimum number of samples required to further split a node.
        lambda: The regularization coefficient for leaf score, also known as lambda.
        gamma: The regularization coefficient for number of tree nodes, also know as gamma.
        learning_rate: The learning rate of gradient boosting.
        n_estimators: Number of trees.
     */
    var estimators:Array[TreeNode] = new Array[TreeNode](n_estimators)
    var score_start:Double=0.0

    def fit(train:DenseMatrix[Double],label:DenseVector[Double]):Unit={
        var tree_constructor = new TreeConstructor(max_depth,min_sample_split,lambda,gamma,train)
        if (loss=="mse"){
            loss_object=mse
        }
        if (loss=="log"){
            loss_object=log
        }
        score_start=mean(label)
        var score = DenseVector.fill(train.rows){score_start}
        for (i:Int <- 0 until n_estimators){
            var estimator=tree_constructor.get_estimator(loss_object.g(score,label),loss_object.h(score,label))
            score+=estimator.predict(train)*:*learning_rate
            estimators(i)=estimator
            println("Building of Tree "+i+" finished")
        }
    }

    def predict(test:DenseMatrix[Double]):DenseVector[Double]={
        var score = DenseVector.fill(test.rows){score_start}
        for (i:TreeNode <- estimators){
            score+=i.predict(test)*:*learning_rate
        }
        loss_object.act(score)
    }

}



class TreeNode(var is_leaf: Boolean=true,
               var score: Double=0.0,
               var feature: Int=0,
               var threshold: Double=0.0,
               var left: TreeNode=null,
               var right: TreeNode=null){
    /*
        The data structure that are used for storing trees.
        A tree is presented by a set of nested TreeNodes,
        with one TreeNode pointing two child TreeNodes,
        until a tree leaf is reached.

        Parameters:
        ----------
        is_leaf: If is TreeNode is a leaf.
        score: The prediction (score) of a tree leaf.
        feature: The split feature of a tree node.
        threshold: The split threshold of a tree node.
        left: Pointing to a child TreeNode,
            where the value of split feature is less than the split threshold.
        right: Pointing to a child TreeNode,
            where the value of split features is greater than or equal to the split threshold.
     */
    def predict(test:DenseMatrix[Double]):DenseVector[Double]={
        test(*,::).map(predict_single)
    }

    def predict_single(test:DenseVector[Double]):Double={
        if (is_leaf){
            return score
        }
        if (test(feature)<threshold){
            return left.predict_single(test)
        } else {
            return right.predict_single(test)
        }
    }
}



class TreeConstructor(var max_depth: Int=3,
           var min_sample_split: Int=10,
           var lambda: Double=1.0,
           var gamma: Double=0,
           var train:DenseMatrix[Double]){
    /*
        This is the class for generating trees.

        Parameters:
        ----------
        max_depth: The maximum depth of the tree.
        min_sample_split: The minimum number of samples required to further split a node.
        lambda: The regularization coefficient for leaf prediction, also known as lambda.
        gamma: The regularization coefficient for number of TreeNode, also know as gamma.
        train: The train matrix

        Methods:
        ----------
        get_estimator: takes g and h and returns a trained tree.
    */
    var g:DenseVector[Double]=null
    var h:DenseVector[Double]=null
    def get_estimator(vector_g:DenseVector[Double],vector_h:DenseVector[Double]):TreeNode={
        g=vector_g
        h=vector_h
        construct_tree(DenseVector.fill(train.rows){true},max_depth)
    }

    def leaf_score(index:DenseVector[Boolean]): Double = {
        -sum(g(index)) / (sum(h(index)) + lambda)
    }

    def leaf_loss(index:DenseVector[Boolean]): Double = {
        -0.5 * pow(sum(g(index)), 2) / (sum(h(index)) + lambda)
    }

    def find_threshold(feature:Int,index:DenseVector[Boolean]): (Double, Double) = {
        var loss = leaf_loss(index)
        var threshold_set = train(index,feature).toArray.toSet.toList.sorted
        var best_gain = 0.0
        var best_threshold = 0.0
        for (i: Int <- 1 until threshold_set.length) {
            var threshold:Double = (threshold_set(i - 1) + threshold_set(i - 1)) / 2
            var left_index =  index &:& (train(::,feature)<:<threshold).toDenseVector
            var right_index = index &:& (train(::,feature)>:=threshold).toDenseVector
            var gain = loss - leaf_loss(left_index) - leaf_loss(right_index)
            if (gain > best_gain) {
                best_threshold = threshold
                best_gain = gain
            }
        }
        (best_threshold, best_gain)
    }

    def find_best_split(index:DenseVector[Boolean]): (Int, Double, Double) = {
        var features=(0 until train.cols).toList.par.map(find_threshold(_,index))

        var best_feature = 0
        var best_threshold = 0.0
        var best_gain = 0.0

        for (i: Int <- 0 until train.cols) {
            var (threshold, gain) = features(i)
            if (gain > best_gain) {
                best_feature = i
                best_threshold = threshold
                best_gain = gain
            }
        }
        (best_feature, best_threshold, best_gain)
    }


    def construct_tree(index: DenseVector[Boolean],depth: Int):TreeNode = {
        /*
        Construct tree recursively.
        First we should check if we should stop further splitting.
        The stopping conditions include:
        1. We have reached the pre-determined max_depth
        2. The number of sample points at this node is less than min_sample_split
        3. The best gain is less than gamma.
        4. Targets take only one value.
        5. Each feature takes only one value.
        By careful design, we could avoid checking condition 4 and 5 explicitly.
        In function find_threshold(), the best_gain is set to 0 initially.
        So if there are no further feature to split,
        or all the targets take the same value,
        the return value of best_gain would be zero.
        Thus condition 3 would be satisfied,
        and no further splitting would be done.
        To conclude, we need only to check condition 1,2 and 3
         */
        if (depth==0 || sum(index.map(if(_) 1 else 0))<min_sample_split){
            return new TreeNode(is_leaf=true,score=leaf_score(index))
        }

        var (best_feature,best_threshold,best_gain) = find_best_split(index)
        if (best_gain<=gamma){
            return new TreeNode(is_leaf=true,score=leaf_score(index))
        }

        var left_index=index &:& (train(::,best_feature)<:<best_threshold).toDenseVector
        var right_index=index &:& (train(::,best_feature)>:=best_threshold).toDenseVector
        var left_child=construct_tree(left_index,depth-1)
        var right_child=construct_tree(right_index,depth-1)
        new TreeNode(is_leaf=false,feature=best_feature,threshold=best_threshold,left=left_child,right=right_child)
    }

}
