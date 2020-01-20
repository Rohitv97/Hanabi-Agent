package com.fossgalaxy.games.fireworks.ai;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class TrainAndSaveModel
{
    private static final int FEATURES_COUNT = 14;
    private static final int CLASSES_COUNT = 1;
    private static final int targetIndex = 14;
    private static final int nEpochs = 20;
    private static final int batchSize = 100;
    private static final boolean saveModel = true;

    public static void main(String[] args) throws IOException, InterruptedException {
        RecordReader trainReader = new CSVRecordReader(1, ',');
        RecordReader testReader = new CSVRecordReader(1, ',');

        trainReader.initialize(new FileSplit(new ClassPathResource("data_g10k.csv").getFile()));
        testReader.initialize(new FileSplit(new ClassPathResource("data_g1k.csv").getFile()));

        //access the data
        DataSetIterator trainIt = new RecordReaderDataSetIterator.Builder(trainReader, batchSize).regression(targetIndex).build();
        DataSetIterator testIt = new RecordReaderDataSetIterator.Builder(testReader, batchSize).regression(targetIndex).build();

        // normalisation
        NormalizerStandardize normalizer = new NormalizerStandardize();
        normalizer.fit(trainIt);
        trainIt.setPreProcessor(normalizer);
        testIt.setPreProcessor(normalizer);

        //build model
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.01))
                .l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(10).build())
                .layer(1, new DenseLayer.Builder().nIn(10).nOut(8).build())
                .layer(2, new OutputLayer.Builder(
                        LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(8).nOut(CLASSES_COUNT).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        model.addListeners(new ScoreIterationListener(100));

        model.fit(trainIt, nEpochs);

        //trainIt.reset();
        //RegressionEvaluation trainEval = model.evaluateRegression(trainIt);
        RegressionEvaluation testEval = model.evaluateRegression(testIt);

        //System.out.println("Training data stats:");
        //System.out.println(trainEval.stats());
        System.out.println("Test data stats:");
        System.out.println(testEval.stats());

        if(saveModel)
        {
            //Save the model
            File locationToSave = new File("Model0.zip");
            ModelSerializer.writeModel(model, locationToSave, false, normalizer);
            System.out.println("Model saved");
        }
    }
}
