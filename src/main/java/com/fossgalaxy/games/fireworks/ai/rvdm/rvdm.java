package com.fossgalaxy.games.fireworks.ai.rvdm;

import com.fossgalaxy.games.fireworks.ai.Agent;
import com.fossgalaxy.games.fireworks.annotations.AgentBuilderStatic;
import com.fossgalaxy.games.fireworks.utils.AgentUtils;

public class rvdm
{
    @AgentBuilderStatic("rvdm")
    public static Agent buildRVDM()
    {
        return AgentUtils.buildAgent("custom_mcs[iggi]");
    }
}
