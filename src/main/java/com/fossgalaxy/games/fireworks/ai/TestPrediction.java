package com.fossgalaxy.games.fireworks.ai;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public class TestPrediction
{
    public static void main(String[] args) throws IOException
    {
        File modelPath = new File("Model0.zip");
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelPath, false);
        NormalizerStandardize normalizer = ModelSerializer.restoreNormalizerFromFile(modelPath);

        //X = 3,0,1,17,6,0,0,0,0,5,0,1,0,0
        //y = 7
        double[][] input = {{3,0,1,17,6,0,0,0,0,5,0,1,0,0}};
        final INDArray input1 = Nd4j.create(input);
        normalizer.transform(input1);
        INDArray out = model.output(input1, false);
        double d = out.getDouble(0);
        System.out.println(d);
        System.out.println(Math.round(d));
    }
}
