package com.fossgalaxy.games.fireworks.ai.rvdm;

import com.fossgalaxy.games.fireworks.state.CardColour;
import com.fossgalaxy.games.fireworks.state.GameState;
import com.fossgalaxy.games.fireworks.state.actions.*;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class Utils
{
    private ArrayList<ArrayList<String>> gameData;

    private void UpdateGameData(GameState state, Action actionToTake)
    {
        ArrayList<String> list = new ArrayList<String>();

        //save data
        list.add(Integer.toString(state.getScore())); //score
        list.add(Integer.toString(state.getInfomation())); //info_tokens
        list.add(Integer.toString(state.getLives())); //lives
        list.add(Integer.toString(state.getDeck().getCardsLeft())); //deck
        list.add(Integer.toString(state.getMovesLeft())); //moves_left

        HashMap<Integer, Integer> counts = GetCardCount(state);
        list.add(Integer.toString(counts.get(5))); //fives
        list.add(Integer.toString(counts.get(4))); //fours
        list.add(Integer.toString(counts.get(3))); //threes
        list.add(Integer.toString(counts.get(2))); //twos
        list.add(Integer.toString(state.getPlayerCount())); //num_players

        list.add(actionToTake instanceof DiscardCard ? "1" : "0"); //discard
        list.add(actionToTake instanceof PlayCard ? "1" : "0"); //play
        list.add(actionToTake instanceof TellColour ? "1" : "0"); //tell_colour
        list.add(actionToTake instanceof TellValue ? "1" : "0"); //tell_value

        gameData.add(list);
    }

    private HashMap<Integer, Integer> GetCardCount(GameState state)
    {
        HashMap<Integer, Integer> result = new HashMap<>();
        result.put(2, 0);
        result.put(3, 0);
        result.put(4, 0);
        result.put(5, 0);

        for(CardColour colour : CardColour.values())
        {
            int val = state.getTableValue(colour);
            for(int i=val ; i>1 ; i--)
            {
                result.put(i, result.get(i) + 1);
            }
        }

        return result;
    }

    private void SaveGameData(int score)
    {
        FileWriter csvOut = null;
        try
        {
            csvOut = new FileWriter("data1.csv", true);

            for(ArrayList<String> innerList : this.gameData)
            {
                innerList.add(Integer.toString(score));

                csvOut.append(String.join(",", innerList));
                csvOut.append("\n");
            }

            this.gameData.clear();
            csvOut.flush();
            csvOut.close();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }
}
