package com.fossgalaxy.games.fireworks.ai.rvdm;

import com.fossgalaxy.games.fireworks.ai.Agent;
import com.fossgalaxy.games.fireworks.ai.rvdm.MCTSPredictor;
import com.fossgalaxy.games.fireworks.annotations.AgentConstructor;
import com.fossgalaxy.games.fireworks.state.Card;
import com.fossgalaxy.games.fireworks.state.GameState;
import com.fossgalaxy.games.fireworks.state.actions.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Vector;

import static com.fossgalaxy.games.fireworks.state.CardColour.*;

/**
 * Flat MC search player
 * <p>
 * Created by piers on 20/12/16.
 */
public class CustomMCS extends MCTSPredictor
{
    /**
     * Constructs a new MC Search player with the given policy to use
     *
     * @param policy The policy to use instead of Random rollouts
     */
    @AgentConstructor("custom_mcs")
    public CustomMCS(Agent policy) throws IOException
    {
        super(new Agent[]{policy, policy, policy, policy, policy});

        // Load the model and normalizer
        File modelPath = new File("Model0.zip");
        this.model = ModelSerializer.restoreMultiLayerNetwork(modelPath, false);
        this.normalizer = ModelSerializer.restoreNormalizerFromFile(modelPath);
        System.out.println("Model loaded");
    }

    @Override
    public Action doMove(int agentID, GameState state)
    {
        return doSuperMove(agentID, state);
    }

    @Override
    protected Action selectActionForRollout(GameState state, int agentID) {
        return agents[agentID].doMove(agentID, state.getCopy());
    }

    @Override
    protected int calculateTreeDepthLimit(GameState state)
    {
        // Experiment with different values
        return 1;
    }

    @Override
    public String toString() {
        return String.format("MCTS(%s)", agents[0].toString());
    }
}