
def comparar_predicions(y_pred, y_val):
    import pandas as pd
    comparar = pd.DataFrame(list(zip(y_pred, y_val)), columns=['y_pred','y_val'])
    comparar["correct"] = comparar.apply(lambda x: 1 if x.y_pred==x.y_val else 0, axis=1)
    return (f"{round((comparar.correct.mean())*100, 3)}% d'accert")
    #return comparar