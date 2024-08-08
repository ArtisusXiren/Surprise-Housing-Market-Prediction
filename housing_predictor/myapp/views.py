from django.shortcuts import render
from django.http import JsonResponse
from .ml_utlis import load_model,pre_process,predict 

xgb_model, rf_model, mappings=load_model()
def predict_view(request):
    if request.method == 'POST':
        input_data=request.POST.get('input_data')
        if input_data:
            input_data=list(input_data.split(','))
            df=pre_process(mappings,input_data)
            xgb_prediction = predict(xgb_model,df)
            rf_prediction = predict(rf_model,df)
            return JsonResponse({
            'xgb_prediction':xgb_prediction.tolist(),
            'rf_prediction': rf_prediction.tolist()
            })
    return render(request,'predict.html')
# Create your views here.
