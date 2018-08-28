package deepdsl.derivation

import deepdsl.analysis._
import deepdsl.ast.{Lmdb, Mnist, _}
import deepdsl.derivation.MemoryAnalysis._
import deepdsl.lang._
import deepdsl.layer._
import deepdsl.run._
import org.junit.Test

class myNet{
	val K = 10
	val path = "deepdsl/gen"

	val batch_size = 256
	val dim = List(batch_size, 1, 28, 28)  // channel, height, width

	val learn_rate = 0.005f
	val momentum = 0.9f
	val decay = 0.0005f

	@Test
	def testLenet = lenet(40, 5, "myLenet")

	private def lenet(train_iter: Int, test_iter: Int, name: String){
		val cv1 = CudaLayer.convolv("cv1", 5, 20)
		val cv2 = CudaLayer.convolv("cv2", 5, 50)
		val mp = CudaLayer.max_pool(2)
		val flat = Layer.flatten(4, 1)
		val fc1 = Layer.full("fc1", 500)
		val fc2 = Layer.full("fc2", K)
		val softmax = CudaLayer.softmax
		val relu = CudaLayer.relu(4)
		val relu2 = CudaLayer.relu(2)

		val network = softmax o fc2 o relu2 o fc1 o flat o mp o relu o cv2 o mp o relu o cv1

		println(typeof(network))

		val x = T._new("X", dim)
		val y = T._new("Y", List(batch_size))
		val xCuda = x.asCuda
		val yCuda = y.asIndicator(K).asCuda

		val loss = (Layer.log_loss(yCuda) o network)(xCuda)
		val accuracy = network(xCuda)

		val param = loss.freeParam.toList
		val solver = Train(name, train_iter, test_iter, learn_rate, momentum, decay, 0)

    	val mnist = Mnist(dim)
    	val loop = Loop(loss, accuracy, mnist, (x, y), param, solver)

    	// runtimeMemory(loop.train.lst)
	    // parameterMemory(param, momentum)
		// generate training and testing file
		CudaCompile(path).print(loop)
	}
}
