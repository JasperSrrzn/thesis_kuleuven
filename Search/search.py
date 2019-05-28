from problog.engine import DefaultEngine
from problog.formula import LogicFormula
from problog.logic import *
from problog.program import PrologString
from data_processing import clear_data_part
from classifiers import SimpleClassifier
import pandas as pd
import problog
import numpy as np
import json

def get_embedding_action(action,predicates,embeddings_enta,embeddings_entb,embeddings_rel,embeddings_rule,randenta,randentb):
    """
    function that concatenates the embeddings given the action = relation, entity a, entity b and rule
    """
    node = action[1]
    functor = db.get_node(node).functor
    for pred in predicates:
        if pred.functor == functor:
            enta = term2str(pred.args[0])
            entb = term2str(pred.args[1])
            if enta.find('ent')==-1:
                enta = randenta
                enta_emb = embeddings_enta[embeddings_enta['entity a']==enta]
                enta_emb = enta_emb.drop(['entity a'],axis=1)
                enta_emb = enta_emb.values.tolist()[0]
            else:
                enta = int(enta[enta.find('ent')+3:])
                enta_emb = embeddings_enta[embeddings_enta['entity a']==enta]
                enta_emb = enta_emb.drop(['entity a'],axis=1)
                enta_emb = enta_emb.values.tolist()[0]

            if entb.find('ent')==-1:
                entb = randentb
                entb_emb = embeddings_entb[embeddings_entb['entity b']==entb]
                entb_emb = entb_emb.drop(['entity b'],axis=1)
                entb_emb = entb_emb.values.tolist()[0]
            else:
                entb = int(entb[entb.find('ent')+3:])
                entb_emb = embeddings_entb[embeddings_entb['entity b']==entb]
                entb_emb = entb_emb.drop(['entity b'],axis=1)
                entb_emb = entb_emb.values.tolist()[0]


    rule_id = db.get_node(node).args[-1]
    rel_id = int(functor[functor.find('rel')+3:])
    rule_emb = embeddings_rule[(embeddings_rule['rule id']==rule_id) & (embeddings_rule['rel id']==rel_id)]
    base = int(rule_emb['base'])
    rule_emb = rule_emb.drop(['rel id','rule id','base'],axis=1)
    rule_emb = rule_emb.values.tolist()[0]
    rel_emb = embeddings_rel[embeddings_rel['rel id']==rel_id]
    rel_emb = rel_emb.drop(['rel id'],axis=1)
    rel_emb = rel_emb.values.tolist()[0]
    features = enta_emb + entb_emb + rel_emb + rule_emb
    return [rel_id] + [enta] + [entb] + [rule_id] + features, base

def rerank_actions(actions,predicates,embeddings_enta,embeddings_entb,embeddings_rel,embeddings_rule,model,neighbors,rand_dict):
    actions_to_rerank = []
    reranked_actions = []
    pos = 0
    for act in actions:
        node = act[1]
        db_term = term2str(db.get_node(node))
        if (db.get_node(node).functor,db.get_node(node).args[:-1]) in rand_dict.keys() and db_term[:db_term.find('(')]=='clause':
            randenta = rand_dict[(db.get_node(node).functor,db.get_node(node).args[:-1])][0]
            randentb = rand_dict[(db.get_node(node).functor,db.get_node(node).args[:-1])][-1]
        else:
            used_preds = []
            pred_pos = 0
            for pred in predicates:
                if pred.functor == db.get_node(node).functor and pred_pos not in used_preds:
                    pred_pos+=1
                    enta = term2str(pred.args[0])
                    entb = term2str(pred.args[1])
                    if enta.find('ent')==-1 and entb.find('ent')!=-1:
                        entb = int(entb[entb.find('ent')+3:])
                        randentb = entb
                        randenta = neighbors[str(entb)][np.random.permutation(len(neighbors[str(entb)]))[0]]
                        rand_dict[(db.get_node(node).functor,db.get_node(node).args[:-1])]=[randenta,randentb]
                    elif entb.find('ent')==-1 and enta.find('ent')!=-1:
                        enta = int(enta[enta.find('ent')+3:])
                        randenta = enta
                        randentb = neighbors[str(enta)][np.random.permutation(len(neighbors[str(enta)]))[0]]
                        rand_dict[(db.get_node(node).functor,db.get_node(node).args[:-1])]=[randenta,randentb]
                    elif enta.find('ent')!=-1 and entb.find('ent')!=-1:
                        randenta = int(enta[enta.find('ent')+3:])
                        randentb = int(entb[entb.find('ent')+3:])
                        rand_dict[(db.get_node(node).functor,db.get_node(node).args[:-1])]=[randenta,randentb]
                    else:
                        randenta = np.random.permutation(14541)[0]
                        randentb = np.random.permutation(14541)[0]

        db_term = term2str(db.get_node(node))
        if db_term[:db_term.find('(')]=='clause':
            data, base = get_embedding_action(act,predicates,embeddings_enta,embeddings_entb,embeddings_rel,embeddings_rule,randenta,randentb)
            actions_to_rerank.append([pos,base]  + data)
        pos+=1

    Reranked_Meta = pd.DataFrame([])
    if len(actions_to_rerank)>1:
        actions_to_rerank = pd.DataFrame(actions_to_rerank)
        Meta = actions_to_rerank[[0,1,2,3,4,5]]
        Data = actions_to_rerank.drop([0,1,2,3,4,5],axis=1)
        predictions = model.predict_proba('./classification_models/model_transe_100/',Data)
        heuristic = predictions[:,-1]
        Meta['heuristic'] = heuristic
        Meta = Meta.rename(columns={0:'position',1:'base',2:'rel id',3:'ent a',4:'ent b',5:'rule id'})
        uniques = []
        for meta_i in range(1,len(Meta)):
            row = Meta.iloc[meta_i]
            if [row['rel id'],row['ent a'], row['ent b']] not in uniques:
                uniques.append([row['rel id'],row['ent a'], row['ent b']])
        for unique in uniques:
            rel_id = unique[0]
            enta = unique[1]
            entb = unique[2]
            Selected = Meta[(Meta['rel id']==rel_id) & (Meta['ent a']== enta) & (Meta['ent b']==entb)]
            Selected_base = Selected[Selected['base']==1]
            Selected_no_base = Selected[Selected['base']==0]
            Selected_base = Selected_base.sort_values(by=['heuristic'])
            Selected_no_base = Selected_no_base.sort_values(by=['heuristic'])
            Reranked_Meta = Reranked_Meta.append(Selected_no_base)
            Reranked_Meta = Reranked_Meta.append(Selected_base)
        act_counter = 0
        for act_i in range(0,len(actions)):
            if act_i not in Reranked_Meta['position']:
                reranked_actions.append(actions[act_i])
            else:
                reranked_actions.append(actions[Reranked_Meta.iloc[act_counter]['position']])
                act_counter+=1

        return reranked_actions, rand_dict
    else:
        return actions,rand_dict


embeddings_enta = pd.read_csv('entity_embeddingsa_transe_100.csv')
embeddings_entb = pd.read_csv('entity_embeddingsb_transe_100.csv')
embeddings_rel = pd.read_csv('relation_embeddings_transe_100.csv')
embeddings_rule = pd.read_csv('rule_embeddings_transe_100.csv')
embeddings_rule = embeddings_rule.drop(['Unnamed: 0'],axis=1)

with open('dict_neighbors.json') as json_file:
    neighbors = json.load(json_file)

model = SimpleClassifier()

print('dimension 100 transe')
print('opening facts and rules')
with open('rules_and_facts.txt', 'r') as myfile:
    facts_and_rules = myfile.readlines()

facts_and_rules_Prolog = PrologString("\n".join(facts_and_rules))

test_data = pd.read_csv('test_data_transe_100.csv')
#test_data = test_data[['entity a','entity b','rel id']]
#test_data = test_data.drop_duplicates()

no_rerank_results = pd.read_csv('results_no_rerank.csv')
pd_results = pd.read_csv('results_transe_100_new.csv')
results = list(pd_results['0'])
#results = []
no_rerank_results = list(no_rerank_results['no rerank'])
for test_sample in range(600,len(test_data)):
    if no_rerank_results[test_sample]>30:
        try:
            print('---------------')
            print(str(test_sample)+ ' : '+str(no_rerank_results[test_sample]))
            row = test_data.iloc[test_sample]
            rel_id = row['rel id']
            ent_a = row['entity a']
            ent_b = row['entity b']
            query = Term('q'+str(rel_id),Term('ent'+str(ent_a)),Term('ent'+str(ent_b)))
            # Perform incremental grounding.
            # This is a split-up of the engine.execute method.

            # Initialize the engine with options:
            #   - unbuffered: don't buffer results internally in the nodes (mimic depth-first construction of target)
            #   - rc_first: first process 'result' and 'complete' messages (allows stopping on 'evaluation' message)
            #   - label_all: (optional) label all intermediate nodes with their predicate
            engine = problog.engine.DefaultEngine(unbuffered=True, rc_first=True, label_all=True)

            # Target formula
            #   - keep_all: don't collapse non-probabilistic subformula's => only for visualization
            target = problog.formula.LogicFormula(keep_all=True)

            db = engine.prepare(facts_and_rules_Prolog)

            # Start the incremental grounding.
            # The result is a list of 'evaluation' actions.
            actions = list(reversed(engine.ground_step(db, query, gp=target)))

            i = 0
            # Execute until no more 'evaluation' actions can be performed.
            rand_dict={}
            while actions:
                actions = engine.execute_step(actions, steps=1, target=target, name=(False, query, 'query'))

                # HERE YOU CAN DO WHATEVER YOU WANT WITH THE ACTION LIST

                # Below is just generating some output.
                i += 1

                #print('==== STEP %d ====' % i)


                predicates = []
                # Go through the engine's stack and extract predicate evaluation nodes ('EvalDefine')
                for rec in engine.stack:
                    if type(rec).__name__ == 'EvalDefine':  # TODO: we should also include 'EvalOr'?
                        nodes = set(b for a, b, in rec.results.results)  # 'target' nodes associated with this evaluation node
                        predicates.append(problog.logic.Term(rec.call[0], *rec.call[1]))
                #print(predicates)
                actions,rand_dict = rerank_actions(actions,predicates,embeddings_enta,embeddings_entb,embeddings_rel,embeddings_rule,model,neighbors,rand_dict)

                #for act in actions:
                #    print(db.get_node(act[1]))

                if type(engine.stack[0]).__name__=='EvalDefine':
                    trigger_nodes = set(b for a,b, in engine.stack[0].results.results)

                #print ('Active predicates:')
                active_nodes = set()  # These are the nodes in 'target' that are still active.
                # Go through the engine's stack and extract predicate evaluation nodes ('EvalDefine')
                for rec in engine.stack:
                    if type(rec).__name__ == 'EvalDefine':  # TODO: we should also include 'EvalOr'?
                        nodes = set(b for a, b, in rec.results.results)  # 'target' nodes associated with this evaluation node
                        #print ('\t', problog.logic.Term(rec.call[0], *rec.call[1]), list(nodes))
                        active_nodes |= nodes # union
                #print ('Active nodes:', list(active_nodes))

                # Visualize and print the current logic program.
                #%dotstr target.to_dot(nodeprops={n: 'fillcolor="red"' for n in active_nodes})

                if len(list(trigger_nodes))>0:
                    print(str(test_sample)+ ':solution found ('+str(i)+')')
                    results.append(i)
                    pd_results = pd.DataFrame(results)
                    pd_results.to_csv('/content/gdrive/My Drive/results_transe_100_new.csv',index=False)
                    break

                if i>no_rerank_results[test_sample]:
                    print(str(test_sample)+ ': no improvement')
                    results.append(0)
                    pd_results = pd.DataFrame(results)
                    pd_results.to_csv('/content/gdrive/My Drive/results_transe_100_new.csv',index=False)
                    break


                if len(actions)==0:
                    print(str(test_sample)+ ':solution not found')
                    results.append(-1)
                    pd_results = pd.DataFrame(results)
                    pd_results.to_csv('/content/gdrive/My Drive/results_transe_100_new.csv',index=False)
                    break
        except:
            results.append(-2)
            pd_results.to_csv('/content/gdrive/My Drive/results_transe_100_new.csv',index=False)
            pass
