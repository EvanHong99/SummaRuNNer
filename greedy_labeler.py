from sumeval.metrics.rouge import RougeCalculator

import re
 



class GreedyLabeler(object):
    """
    贪心算法用于提取一篇文章中可以作为摘要的句子
    """
    def __init__(self,sep:str="。!！?？；;，,"):
        self.sep=sep
        self.rouge = RougeCalculator(stopwords=False, lang="zh")

    def sep_ref(self,ref:str)->list:
        reg=rf'[{self.sep}]' 
        # reg=rf'([{self.sep}])' #用来保留分隔符
        splitted=[]
        for s in re.split(reg,ref.strip()):
            if len(s.strip())!=0:
                splitted.append(s)
        # if len(splitted)%2==1:
        #     return ["".join(i).strip() for i in zip(splitted[0::2],splitted[1::2])]+[splitted[-1].strip()] #将标点合并到原句子
        # return ["".join(i).strip() for i in zip(splitted[0::2],splitted[1::2])] #将标点合并到原句子

        return splitted

    def is_newly_best(self,strtgy,max_r1,max_r2,max_rL,new_r1,new_r2,new_rL):
        if strtgy =="rouge_1":
            if new_r1>max_r1: return True
        elif strtgy =="rouge_2":
            if new_r2>max_r2: return True
        elif strtgy =="rouge_L":
            if new_rL>max_rL: return True
        elif strtgy =="rouge_all":
            if (new_r1,new_r2,new_rL)>(max_r1,max_r2,max_rL): return True
        return False


    def label(self,summary:str,ref_list:list,strtgy='rouge_1'):
        """
        按照rouge1最大化的默认规则来构建label
        """
        assert strtgy in ['rouge_1','rouge_2','rouge_L','rouge_all']
        best=[]
        idx=[0]*len(ref_list)
        max_r1=0
        max_r2=0
        max_rL=0
        # 迭代len步
        for j in range(len(ref_list)):
            append_idx=-1
            append=False
            for i,ref in enumerate(ref_list):
                temp=best+[ref]
                new_r1=self.rouge.rouge_n(
                    summary=summary,
                    references=" ".join(temp),
                    n=1)
                new_r2=self.rouge.rouge_n(
                    summary=summary,
                    references=" ".join(temp),
                    n=2)
                new_rL=self.rouge.rouge_l(
                    summary=summary,
                    references=" ".join(temp))
                append=self.is_newly_best(strtgy,max_r1,max_r2,max_rL,new_r1,new_r2,new_rL)            
                if append:
                    max_r1=new_r1
                    max_r2=new_r2
                    max_rL=new_rL
                    append_idx=i
                    best.append(ref_list[append_idx])
                    idx[append_idx]=1
            else:
                break
        return best,idx,tuple([max_r1,max_r2,max_rL])

if __name__=='__main__':
    sum="雅虎 宣布 剥离 阿里巴巴 股份"
    refs="雅虎 发布 2014 年 第四季度 财报 ， 并 推出 了 免税 方式 剥离 其 持有 的 阿里巴巴 集团 15 ％ 股权 的 计划 ， 打算 将 这 一 价值 约 400 亿美元 的 宝贵 投资 分配 给 股东 。 截止 发稿 前 ， 雅虎 股价 上涨 了 大约 7 ％ ， 至 51.45 美元 。"
    gl=GreedyLabeler()
    best,idx,max_score=gl.label(sum,gl.sep_ref(refs))
    print(best)
    print(idx)

