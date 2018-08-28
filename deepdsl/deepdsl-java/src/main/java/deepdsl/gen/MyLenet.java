package deepdsl.gen;
import deepdsl.cudnn.*;
import deepdsl.cudnn.config.*;
import deepdsl.tensor.*;
import deepdsl.util.CudaRun;


public class MyLenet extends CudaRun {

public static void main(String[] args){
MyLenet run = new MyLenet();
run.train(40);
run.test(5);
run.save();
run.free();
}

public MyLenet() {
super("src/main/java/deepdsl/gen/myLenet");
setTrainData(MnistFactory.getFactory(true, new int[]{256, 1, 28, 28}));
setTestData(MnistFactory.getFactory(false, new int[]{256, 1, 28, 28}));
}

float lrn_rate = -0.005f;
float momentum = 0.9f;
float decay = 5.0E-4f;

JCudnnConvolution y8 = addConvolution(new int[]{256,1,28,28},new int[]{20,1,5,5},new int[]{20}, 1, 0);
JCudnnConvolution y5 = addConvolution(new int[]{256,20,12,12},new int[]{50,20,5,5},new int[]{50}, 1, 0);
JCudnnPooling y6 = addPooling(new int[]{256,20,24,24}, 2, 2, 0, PoolingType.MAX);
JCudnnPooling y3 = addPooling(new int[]{256,50,8,8}, 2, 2, 0, PoolingType.MAX);
JCudnnActivation y7 = addActivation(new int[]{256,20,24,24}, ActivationMode.RELU);
JCudnnActivation y4 = addActivation(new int[]{256,50,8,8}, ActivationMode.RELU);
JCudnnActivation y2 = addActivation(new int[]{256,500}, ActivationMode.RELU);
JCudnnSoftmax y1 = addSoftmax(new int[]{256,10}, SoftmaxAlgorithm.ACCURATE);
JCudaTensor V_cv1_B = addParam("V_cv1_B", "Constant", 0f, 20);
JCudaTensor V_cv1_W = addParam("V_cv1_W", "Constant", 0f, 20, 1, 5, 5);
JCudaTensor V_cv2_B = addParam("V_cv2_B", "Constant", 0f, 50);
JCudaTensor V_cv2_W = addParam("V_cv2_W", "Constant", 0f, 50, 20, 5, 5);
JCudaTensor V_fc1_B = addParam("V_fc1_B", "Constant", 0f, 500);
JCudaTensor V_fc1_W = addParam("V_fc1_W", "Constant", 0f, 500, 800);
JCudaTensor V_fc2_B = addParam("V_fc2_B", "Constant", 0f, 10);
JCudaTensor V_fc2_W = addParam("V_fc2_W", "Constant", 0f, 10, 500);
JCudaTensor cv1_B = addParam("cv1_B", "Constant", 0.0f, 20);
JCudaTensor cv1_W = addParam("cv1_W", "Random", 0.28284273f, 20, 1, 5, 5);
JCudaTensor cv2_B = addParam("cv2_B", "Constant", 0.0f, 50);
JCudaTensor cv2_W = addParam("cv2_W", "Random", 0.06324555f, 50, 20, 5, 5);
JCudaTensor fc1_B = addParam("fc1_B", "Constant", 0.0f, 500);
JCudaTensor fc1_W = addParam("fc1_W", "Random", 0.05f, 500, 800);
JCudaTensor fc2_B = addParam("fc2_B", "Constant", 0.0f, 10);
JCudaTensor fc2_W = addParam("fc2_W", "Random", 0.06324555f, 10, 500);

public float trainFunction(JTensorFloat X, JTensorFloat Y) {
 // val X77 = Cuda(X)
JCudaTensor X77 = X.asJCudaTensor();
// val X80 = Cuda(Indicator(Y, 10))
JCudaTensor X80 = Y.asIndicator(10).asJCudaTensor();
// val X27 = Convolv(1,0)(X77,cv1_W,cv1_B)
JCudaTensor X27 = y8.forward(X77, cv1_W, cv1_B);
// val X28 = ReLU()(X27)
JCudaTensor X28 = y7.forward(X27);
// val X29 = Pooling(2,2,0,true)(X28)
JCudaTensor X29 = y6.forward(X28);
// val X30 = Convolv(1,0)(X29,cv2_W,cv2_B)
JCudaTensor X30 = y5.forward(X29, cv2_W, cv2_B);
// val X31 = ReLU()(X30)
JCudaTensor X31 = y4.forward(X30);
// val X32 = Pooling(2,2,0,true)(X31)
JCudaTensor X32 = y3.forward(X31);
// val X75 = (X32[1><3])(i10 | @) * (fc1_W)(i11 | @)
JCudaTensor X75 = X32.flatten(1, new int[]{50, 4, 4}).asMatrix(1, true).times(fc1_W.asMatrix(1, true));
// val X34 = (X75 + (i10) => fc1_B)
JCudaTensor X34 = fc1_B.copy(256, X75);
// val X35 = ReLU()(X34)
JCudaTensor X35 = y2.forward(X34);
// val X78 = (X35)(i13 | @) * (fc2_W)(i14 | @)
JCudaTensor X78 = X35.asMatrix(1, true).times(fc2_W.asMatrix(1, true));
// val X36 = (X78 + (i13) => fc2_B)
JCudaTensor X36 = fc2_B.copy(256, X78);
// val X37 = Softmax()(X36)
JCudaTensor X37 = y1.forward(X36);
// dealloc X36
X36.free();
// val X81 = Log X37.copy
JCudaTensor X81 = X37.clone().log();
// val _loss = ((0 - (X80 . X81)) / |256|)
float _loss = - X80.dot(X81) / 256f;
// dealloc X81
X81.free();
// val m3 = (i40) => fc2_W[@, i40]
JCudaMatrix m3 = fc2_W.asMatrix(1, false);
// val m6 = (i19) => X35[@, i19]
JCudaMatrix m6 = X35.asMatrix(1, false);
// val X83 = 1/(X37.copy)
JCudaTensor X83 = X37.clone().pow(-1f);
// val X84 = X80.copy .* X83
JCudaTensor X84 = X80.clone().times_i(X83);;
// dealloc X80
X80.free();
// dealloc X83
X83.free();
// val X85 = - X84
JCudaTensor X85 = X84.times_i(-1f);;
// val X38 = (X85 / |256|)
JCudaTensor X38 = X85.times_i(1 / 256f);;
// val X109 = X38 * d_Softmax()(X37)/d_X36
JCudaTensor X109 = y1.backward(X38, X37);
// dealloc X38
X38.free();
// dealloc X37
X37.free();
// val m2 = (i43) => X109[@, i43]
JCudaMatrix m2 = X109.asMatrix(1, false);
// V_fc2_W = ((m2 * m6 * -0.005) + (V_fc2_W * 0.9))
m2.times(m6, V_fc2_W, lrn_rate, momentum);
// fc2_W = (V_fc2_W + (fc2_W * (1 + (5.0E-4 * -0.005))))
fc2_W.update(V_fc2_W, 1f, 1f + decay * lrn_rate);
// V_fc2_B = ((Sum(m2) * -0.005) + (V_fc2_B * 0.9))
m2.sum(V_fc2_B, lrn_rate, momentum);
// fc2_B = (V_fc2_B + (fc2_B * (1 + (5.0E-4 * -0.005))))
fc2_B.update(V_fc2_B, 1f, 1f + decay * lrn_rate);
// val X96 = (X109)(i39 | @) * m3
JCudaTensor X96 = X109.asMatrix(1, true).times(m3);
// dealloc X109
X109.free();
// val X104 = X96 * d_ReLU()(X35)/d_X34
JCudaTensor X104 = y2.backward(X96, X35);
// dealloc X35
X35.free();
// val m1 = (i36) => X104[@, i36]
JCudaMatrix m1 = X104.asMatrix(1, false);
// V_fc1_B = ((Sum(m1) * -0.005) + (V_fc1_B * 0.9))
m1.sum(V_fc1_B, lrn_rate, momentum);
// fc1_B = (V_fc1_B + (fc1_B * (1 + (5.0E-4 * -0.005))))
fc1_B.update(V_fc1_B, 1f, 1f + decay * lrn_rate);
// val m4 = (i33) => fc1_W[@, i33]
JCudaMatrix m4 = fc1_W.asMatrix(1, false);
// val m8 = (i23) => X32[1><3][@, i23]
JCudaMatrix m8 = X32.flatten(1, new int[]{50, 4, 4}).asMatrix(1, false);
// V_fc1_W = ((m1 * m8 * -0.005) + (V_fc1_W * 0.9))
m1.times(m8, V_fc1_W, lrn_rate, momentum);
// fc1_W = (V_fc1_W + (fc1_W * (1 + (5.0E-4 * -0.005))))
fc1_W.update(V_fc1_W, 1f, 1f + decay * lrn_rate);
// val X97 = (X104)(i32 | @) * m4
JCudaTensor X97 = X104.asMatrix(1, true).times(m4);
// dealloc X104
X104.free();
// val X95 = X97[1<>3] * d_Pooling(2,2,0,true)(X32,X31)/d_X31
JCudaTensor X95 = y3.backward(X97.unflatten(1, new int[]{50, 4, 4}), X32, X31);
// dealloc X32
X32.free();
// dealloc X97
X97.free();
// val X107 = X95 * d_ReLU()(X31)/d_X30
JCudaTensor X107 = y4.backward(X95, X31);
// dealloc X31
X31.free();
// val X102 = X107 * d_Convolv(1,0)(cv2_W)/d_X29
JCudaTensor X102 = y5.backward_data(X107, cv2_W);
// V_cv2_W = ((X107 * d_Convolv(1,0)(X29)/d_cv2_W * -0.005) + (V_cv2_W * 0.9))
y5.backward_filter(X107, X29, V_cv2_W, lrn_rate, momentum);
// cv2_W = (V_cv2_W + (cv2_W * (1 + (5.0E-4 * -0.005))))
cv2_W.update(V_cv2_W, 1f, 1f + decay * lrn_rate);
// V_cv2_B = ((X107 * d_Convolv(1,0)()/d_cv2_B * -0.005) + (V_cv2_B * 0.9))
y5.backward_bias(X107, V_cv2_B, lrn_rate, momentum);
// dealloc X107
X107.free();
// cv2_B = (V_cv2_B + (cv2_B * (1 + (5.0E-4 * -0.005))))
cv2_B.update(V_cv2_B, 1f, 1f + decay * lrn_rate);
// val X99 = X102 * d_Pooling(2,2,0,true)(X29,X28)/d_X28
JCudaTensor X99 = y6.backward(X102, X29, X28);
// dealloc X29
X29.free();
// dealloc X102
X102.free();
// val X101 = X99 * d_ReLU()(X28)/d_X27
JCudaTensor X101 = y7.backward(X99, X28);
// dealloc X28
X28.free();
// V_cv1_W = ((X101 * d_Convolv(1,0)(X77)/d_cv1_W * -0.005) + (V_cv1_W * 0.9))
y8.backward_filter(X101, X77, V_cv1_W, lrn_rate, momentum);
// dealloc X77
X77.free();
// cv1_W = (V_cv1_W + (cv1_W * (1 + (5.0E-4 * -0.005))))
cv1_W.update(V_cv1_W, 1f, 1f + decay * lrn_rate);
// V_cv1_B = ((X101 * d_Convolv(1,0)()/d_cv1_B * -0.005) + (V_cv1_B * 0.9))
y8.backward_bias(X101, V_cv1_B, lrn_rate, momentum);
// dealloc X101
X101.free();
// cv1_B = (V_cv1_B + (cv1_B * (1 + (5.0E-4 * -0.005))))
cv1_B.update(V_cv1_B, 1f, 1f + decay * lrn_rate);

return _loss; 
}

public JCudaTensor testFunction(JTensorFloat X) {
 // val X11 = Cuda(X)
JCudaTensor X11 = X.asJCudaTensor();
// val X12 = Convolv(1,0)(X11,cv1_W,cv1_B)
JCudaTensor X12 = y8.forward(X11, cv1_W, cv1_B);
// dealloc X11
X11.free();
// val X13 = ReLU()(X12)
JCudaTensor X13 = y7.forward(X12);
// val X14 = Pooling(2,2,0,true)(X13)
JCudaTensor X14 = y6.forward(X13);
// dealloc X13
X13.free();
// val X15 = Convolv(1,0)(X14,cv2_W,cv2_B)
JCudaTensor X15 = y5.forward(X14, cv2_W, cv2_B);
// dealloc X14
X14.free();
// val X16 = ReLU()(X15)
JCudaTensor X16 = y4.forward(X15);
// val X17 = Pooling(2,2,0,true)(X16)
JCudaTensor X17 = y3.forward(X16);
// dealloc X16
X16.free();
// val X23 = (X17[1><3])(i10 | @) * (fc1_W)(i11 | @)
JCudaTensor X23 = X17.flatten(1, new int[]{50, 4, 4}).asMatrix(1, true).times(fc1_W.asMatrix(1, true));
// dealloc X17
X17.free();
// val X19 = (X23 + (i10) => fc1_B)
JCudaTensor X19 = fc1_B.copy(256, X23);
// val X20 = ReLU()(X19)
JCudaTensor X20 = y2.forward(X19);
// val X25 = (X20)(i13 | @) * (fc2_W)(i14 | @)
JCudaTensor X25 = X20.asMatrix(1, true).times(fc2_W.asMatrix(1, true));
// dealloc X20
X20.free();
// val X21 = (X25 + (i13) => fc2_B)
JCudaTensor X21 = fc2_B.copy(256, X25);
// val X22 = Softmax()(X21)
JCudaTensor X22 = y1.forward(X21);
// dealloc X21
X21.free();

return X22; 
}

}