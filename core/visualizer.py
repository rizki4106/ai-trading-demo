import pandas as pd

def annotate(annotation : list, prediction : list, frame : pd.DataFrame, class_name : dict):

    result = []

    for ann, pred in zip(annotation, prediction):
        
        ant = dict(
                x=frame.index[ann - 1],
                y=frame['High'][ann - 1],
                xref='x',
                yref='y',
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=-40 if pred == 0 else 40,
                text=class_name[pred],
            )

        result.append(ant)

    return result