package com.fossgalaxy.games.fireworks.ai;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;

public class LoadAndTestModel
{
    private static final int targetIndex = 14;
    private static final int batchSize = 100;

    public static void main(String[] args) throws IOException, InterruptedException
    {
        File modelPath = new File("Model0.zip");
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelPath, false);
        NormalizerStandardize normalizer = ModelSerializer.restoreNormalizerFromFile(modelPath);

        RecordReader testReader = new CSVRecordReader(1, ',');
        testReader.initialize(new FileSplit(new ClassPathResource("data_g1k.csv").getFile()));
        DataSetIterator testIt = new RecordReaderDataSetIterator.Builder(testReader, batchSize).regression(targetIndex).build();
        testIt.setPreProcessor(normalizer);

        RegressionEvaluation testEval = model.evaluateRegression(testIt);
        System.out.println("Test data stats:");
        System.out.println(testEval.stats());
    }
}
