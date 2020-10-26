import torch
import torch.nn as nn
import torch.nn.functional as F
#model里不再分别设置state、target之类的概念，而是要对input中的每一个成分进行处理
#target可能有不同成分，以后修复
class LstmModel(torch.nn.Module):
    """一个简单的模型,model里不再分别设置state、target之类的概念，
    而是要对input中的每一个成分进行处理"""
    def __init__(
        self,
        action_sz,
        state_sz = 2048,
        target_sz = 300,
    ):
        super(LstmModel, self).__init__()
        
        target_embed_sz = 512
        state_embed_sz = 512
        self.embed_state = nn.Linear(state_sz, state_embed_sz)
        self.embed_target = nn.Linear(target_sz, target_embed_sz)
        middle_sz = state_embed_sz + target_embed_sz

        #navigation architecture
        navi_arch_out_sz = 512
        self.navi_net = nn.LSTMCell(middle_sz, navi_arch_out_sz)

        #output
        self.actor_linear = nn.Linear(navi_arch_out_sz, action_sz)
        self.critic_linear = nn.Linear(navi_arch_out_sz, 1)

    def forward(self, model_input, params = None):
        '''保证输入的数据都是torch的tensor'''
        (hx, cx) = model_input['hidden']
        state = model_input['fc']
        target = model_input['glove']
        if params is None:
            state_embed2 = self.embed_state(state)
            state_embed = F.relu(state_embed2, True)
            target_embed = F.relu(self.embed_target(model_input['glove']), True)
            x = torch.cat((state_embed, target_embed), dim=1)

            hx, cx = self.navi_net(x, (hx, cx))
            x = F.relu(hx, True)
            actor_out = self.actor_linear(x)
            critic_out = self.critic_linear(x)
        else:
            state_embed = F.relu(
                F.linear(
                    state,
                    weight=params["embed_state.weight"],
                    bias=params["embed_state.bias"],
                    ), True
                )
            target_embed = F.relu(
                F.linear(
                    target,
                    weight=params["embed_target.weight"],
                    bias=params["embed_target.bias"],
                    ), True
                )
            x = torch.cat((state_embed, target_embed), dim=1)
            hx, cx = torch.lstm_cell(
                x,
                (hx, cx),
                params["navi_net.weight_ih"],
                params["navi_net.weight_hh"],
                params["navi_net.bias_ih"],
                params["navi_net.bias_hh"],
            )
            x = F.relu(hx, True)

            critic_out = F.linear(
                x,
                weight=params["critic_linear.weight"],
                bias=params["critic_linear.bias"],
            )
            actor_out = F.linear(
                x,
                weight=params["actor_linear.weight"],
                bias=params["actor_linear.bias"],
            )

        return dict(
            policy = actor_out,
            value = critic_out,
            hidden = (hx, cx)
            )

if __name__ == "__main__":
    model = LstmModel(4,2048,300)
    input1 = torch.randn(3,2048)
    input2 = torch.randn(1,300)
    print(input2)
    out = model.forward(dict(fc=input1, glove=input2))
    print(out['policy'])
    print(out['value'])

