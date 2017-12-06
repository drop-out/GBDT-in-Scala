import breeze.linalg._
import breeze.numerics._
import breeze.stats._
import java.io.File

object example {
    def main(args: Array[String]): Unit = {

        //读取数据
        var file = new File("data/winequality-red.csv")
        var data = csvread(file,';')

        //分训练集和测试集
        var train = data(0 to 1200,0 to data.cols-2)
        var train_label = data(0 to 1200,-1)
        var test = data(1201 to -1,0 to data.cols-2)
        var test_label = data(1201 to -1,-1)

        //训练
        var model = new GBDT(max_depth=3,learning_rate=0.01,n_estimators=700)
        model.fit(train,train_label)

        //评分
        println(mean(abs(model.predict(test)-test_label)))
    }

}
