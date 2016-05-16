package ov;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

/**
 * Created by Peerachart Piriyarattanachai on 15/5/2559.
 */
public class TestImage {
    private static final Logger log = LoggerFactory.getLogger(TestImage.class);
    public static void main( String[] args ) throws IOException{
        int nChannels = 1;
        int outputNum = 2;
        int batchSize = 25;
        int nEpochs = 10;
        int iterations = 1;
        int seed = 123;

        // Load Network Configuration from Disk...
        MultiLayerConfiguration conf = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File("conf.json")));
        // Load Parameter from disk
        INDArray newParams;
        try(DataInputStream dis = new DataInputStream(new FileInputStream("coefficients.bin"))){
            newParams = Nd4j.read(dis);
        }

        // **** ตรงนี้ต้องเปลี่ยนเป็นออะไรอ่ะพี่ อาร์มต้องการ
        // 1. เอา model ที่ได้จากการ train มาแล้วมาเทสด้วยภาพต่างๆ เพื่อแยกว่าภาพที่ใส่ไปใช่ไข่หรือไม่
        // 2. เพิ่ม path  ที่เอาไฟลล์เขาไปเทสกับ model ตรงไหน code ไหน
        // 3. ตรง Evaluate ก็จะเป็น Accuracy ของการเทสกับภาพต่างๆ
        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);


        // Create MultilayerNetwork from the saved configuration and Parameters
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setParameters(newParams);

        /* log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.01)//.biasLearningRate(0.02)
                //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(20)
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nIn(nChannels)
                        .stride(1, 1)
                        .nOut(50)
                        .activation("identity")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder().activation("relu")
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,28,28,1);


        */

        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for( int i=0; i<nEpochs; i++ ) {
            model.fit(mnistTrain);
            log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while(mnistTest.hasNext()){
                DataSet ds = mnistTest.next();
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);
            }
            log.info(eval.stats());
            mnistTest.reset();
        }
        log.info("****************Example finished********************");
    }
}
