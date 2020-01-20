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
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.File;
import java.io.IOException;

public class TrainNet
{
    private static final int FEATURES_COUNT = 14;
    private static final int CLASSES_COUNT = 1;
    public static final int nEpochs = 100;

    public static void main(String[] args)
    {
        try (RecordReader recordReader = new CSVRecordReader(1, ','))
        {
            //Read the csv file
            recordReader.initialize(new FileSplit(
                    new ClassPathResource("data.csv").getFile()));

            //access the data and shuffle it
            DataSetIterator iterator = new RecordReaderDataSetIterator(
                    recordReader, 30, FEATURES_COUNT, CLASSES_COUNT);
            DataSet allData = iterator.next();
            allData.shuffle(42);

            //normalize data
            //DataNormalization normalizer = new NormalizerStandardize();
            //normalizer.fit(allData);
            //normalizer.transform(allData);

            //split data into test and train
            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.80);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testData = testAndTrain.getTest();

            //build model
            MultiLayerConfiguration configuration
                    = new NeuralNetConfiguration.Builder()
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Nesterovs(0.1, 0.9))
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

            for(int i = 0; i < nEpochs; i++)
            {
                model.fit(trainingData);
            }

            DataSetIterator mytest = new RecordReaderDataSetIterator(
                    recordReader, 30, FEATURES_COUNT, CLASSES_COUNT);
            RegressionEvaluation eval = model.evaluateRegression(mytest);

            System.out.println(eval.stats());

            //Save the model
            File locationToSave = new File("MyMultiLayerNetwork.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
            boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
            model.save(locationToSave, saveUpdater);

            System.out.println("Saved the model");
            System.out.println("Trying to predict a value: ");

            double[][] input = {{0,8,3,35,4,0,0,0,0,3,0,0,0,1}};
            final INDArray input1 = Nd4j.create(input);
            INDArray out = model.output(input1, false);
            System.out.println(out);

        }
        catch (InterruptedException e)
        {
            e.printStackTrace();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }

    }
}
