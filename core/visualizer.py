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

def create_annotation_v2(annt, result, class_name, data):
    """
    Create annotation for the chart
    Args:
        data : pandas DataFrame -> ohlc data
        annt : list -> annotation point
        result : list -> result from prediction
        class_name : dict -> class name
    Returns:
        annotation : list -> list of dict
    """
    # define annotation variable
    annotation = []

    # loop through annt and result
    for i in range(len(annt)):
        # define dict
        dict_ = dict(x=data.index[annt[i] - 1],
                    y=data['High'][annt[i] - 1],
                    xref='x',
                    yref='y',
                    showarrow=True,
                    arrowhead=7,
                    ax=0,
                    ay=-40 if result[i] == 1 else 40,
                    text=f'{class_name[result[i]]}')
        # append dict to annotation
        annotation.append(dict_)
    return annotation