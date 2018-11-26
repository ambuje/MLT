
from django.views.generic import TemplateView
from django.shortcuts import render , redirect,render_to_response,reverse

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from django.shortcuts import render , HttpResponse
from matplotlib import pylab
from django.http import HttpResponse
from matplotlib import pylab
from pylab import *
import PIL, PIL.Image
from pylab import *
from .forms import IntForm,mlr_regression,polynomial_regression
import numpy as np
from io import StringIO, BytesIO
import PIL.Image
import json



import io
from io import *
from matplotlib import pylab
from django.http import HttpResponseRedirect
from .models import file
from django.views.generic import FormView, DetailView, ListView
from sklearn.linear_model import LinearRegression
import PIL
from mltool.functions.functions import handle_uploaded_file

try:
    def error_404(request):

        return render(request, 'htmls\\documentation.html')


    def error_500(request):

        return render(request, 'htmls\\documentation.html')


    def documentation(request):


        return render(request,'htmls\\documentation.html')

    def index(request):
        return render(request,'htmls\\index.html')




    class linear(TemplateView):

        template_name = 'C:\\Users\\ambuj\\Desktop\\mltool\\templates\\htmls\\lr.html'




        def get(self,request):
            form=IntForm()


            return render(request,'htmls\\lr_get.html',{'form':form})

        def post(self,request):
            form=IntForm()
            if request.method == 'POST' or 'FILES' :

                form=IntForm(request.POST,request.FILES)
                if form.is_valid():


                    handle_uploaded_file(request.FILES['upload'])








                    x1_ind=form.cleaned_data['col_start']

                    y1_dep=form.cleaned_data['col_y']

                    file_name='C:\\Users\\ambuj\\Desktop\\mltool\\mltool\\static\\csv\\' + request.FILES['upload'].name

                    dataset = pd.read_csv(file_name)
                    X = dataset.iloc[:,x1_ind-1].values
                    y = dataset.iloc[:,y1_dep-1].values
                    X = np.array(X).reshape(-1, 1)
                    y = np.array(y).reshape(-1, 1)





                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
                    # Fitting Simple Linear Regression to Training Set

                    regressor = LinearRegression()
                    regressor.fit(X_train, y_train)
                    y_pred = regressor.predict(X_test)




                    text=form.cleaned_data['prediction']





                    y_pred1=regressor.predict(text)
                    form=IntForm()

                a = X_train.tolist()
                b=y_pred.tolist()
                l=[]
                l1=[]
                for i in range(0,len(a)):
                    l.append(a[i][0])
                for i in range(0,len(b)):
                    l1.append(b[i][0])
                am=json.dumps(l)
                ll1=json.dumps(l1)

                plt.scatter(X_train, y_train, color='red')
                plt.plot(X_train, regressor.predict(X_train), color='blue')
                plt.title('Linear Regression')
                plt.xlabel('Independent Variable')
                plt.ylabel('Dependent Variable')

                plt.savefig('mltool/static/csv/train.jpeg')
                plt.gcf().clear()

                plt.scatter(X_test, y_test, color='red')
                plt.plot(X_train, regressor.predict(X_train), color='blue')
                plt.title('Linear Regression')
                plt.xlabel('Independent Variable')
                plt.ylabel('Dependent Variable')
                plt.savefig('mltool/static/csv/test.jpeg')

                plt.gcf().clear()


                args = {'form': form, 'result': y_pred1[0][0],'x_train':l,'y_pred':l1,'ll1':ll1,'am':am}


                return render(request, self.template_name, args)


    class multiple(TemplateView):
        template_name = 'C:\\Users\\ambuj\\Desktop\\mltool\\templates\\htmls\\mlr.html'
        def get(self,request):
            form=mlr_regression()
            return render(request,'htmls\\mlr_get.html',{'form':form})

        def post(self,request):
            form=mlr_regression()
            if request.method == 'POST' or 'FILES' :

                form=mlr_regression(request.POST,request.FILES)
                if form.is_valid():


                    handle_uploaded_file(request.FILES['upload'])








                    x1_ind=form.cleaned_data['col_start']
                    x2_ind=form.cleaned_data['col_end']

                    y1_dep=form.cleaned_data['col_y']

                    file_name='C:\\Users\\ambuj\\Desktop\\mltool\\mltool\\static\\csv\\' + request.FILES['upload'].name

                    dataset = pd.read_csv(file_name)
                    X = dataset.iloc[:,x1_ind-1 : x2_ind -1].values
                    y = dataset.iloc[:,y1_dep-1].values
                    # X = np.array(X).reshape(-1, 1)
                    # y = np.array(y).reshape(-1, 1)

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

                    # Fitting Multiple Linear Regression to the Training set
                    from sklearn.linear_model import LinearRegression
                    regressor = LinearRegression()
                    regressor.fit(X_train, y_train)

                    # Predicting the Test set results
                    y_pred = regressor.predict(X_test)
                    list=[]

                    for i in range(0,3):
                        text=form.cleaned_data['prediction']
                        list.append(text)






                    y_pred1=regressor.predict([list])
                    # y_pred2=regressor.predict()
                    # y_pred3= regressor.predict()
                    form=mlr_regression()

                plt.scatter(X_train[:, 0], y_train, color='red')
                # plt.plot(X_train[:,0], regressor.predict(X_train[:,0]), color = 'blue')
                plt.title('MLR')
                plt.xlabel('Independent Variable 1')
                plt.ylabel('Dependent Variable')
                plt.savefig('mltool/static/csv/train1.jpeg')
                plt.gcf().clear()
                # plt.switch_backend('qt4agg')

                plt.scatter(X_train[:, 1], y_train, color='red')
                # plt.plot(X_train[:,0], regressor.predict(X_train[:,0]), color = 'blue')
                plt.title('MLR')
                plt.xlabel('Independent Variable 2')
                plt.ylabel('Dependent Variable')
                plt.savefig('mltool/static/csv/train2.jpeg')
                plt.gcf().clear()

                plt.scatter(X_train[:, 2], y_train, color='red')
                # plt.plot(X_train[:,0], regressor.predict(X_train[:,0]), color = 'blue')
                plt.title('MLR')
                plt.xlabel('Independent Variable 3')
                plt.ylabel('Dependent Variable')
                plt.savefig('mltool/static/csv/train3.jpeg')
                plt.gcf().clear()

                # Visualising the Test set results
                plt.scatter(X_test[:, 0], y_test, color='red')
                # plt.plot(X_train[:,0], regressor.predict(X_train[:,0]), color = 'blue')
                plt.title('MLR')
                plt.xlabel('Independent Variable 1')
                plt.ylabel('Dependent Variable')
                plt.savefig('mltool/static/csv/test1.jpeg')
                plt.gcf().clear()
                # plt.switch_backend('qt4agg')

                plt.scatter(X_test[:, 1], y_test, color='red')
                # plt.plot(X_train[:,0], regressor.predict(X_train[:,0]), color = 'blue')
                plt.title('MLR')
                plt.xlabel('Independent Variable 2')
                plt.ylabel('Dependent Variable')
                plt.savefig('mltool/static/csv/test2.jpeg')
                plt.gcf().clear()

                plt.scatter(X_test[:, 2], y_test, color='red')
                # plt.plot(X_train[:,0], regressor.predict(X_train[:,0]), color = 'blue')
                plt.title('MLR')
                plt.xlabel('Independent Variable 3')
                plt.ylabel('Dependent Variable')
                plt.savefig('mltool/static/csv/test3.jpeg')
                plt.gcf().clear()





                args = {'form': form, 'result': y_pred1[0]}


                return render(request, self.template_name, args)



    class polynomial(TemplateView):
        template_name = 'C:\\Users\\ambuj\\Desktop\\mltool\\templates\\htmls\\pr.html'
        def get(self,request):
            form=polynomial_regression()
            return render(request,'htmls\\pr_get.html',{'form':form})
        def post(self, request):
            form = polynomial_regression()
            if request.method == 'POST' or 'FILES':

                form = polynomial_regression(request.POST, request.FILES)
                if form.is_valid():
                    handle_uploaded_file(request.FILES['upload'])

                    x1_ind = form.cleaned_data['col_start']
                    x2_ind = form.cleaned_data['col_end']

                    y1_dep = form.cleaned_data['col_y']
                    deg = form.cleaned_data['degree']  #degree ka dabba bnana h

                    file_name = 'C:\\Users\\ambuj\\Desktop\\mltool\\mltool\\static\\csv\\' + request.FILES['upload'].name

                    dataset = pd.read_csv(file_name)
                    if(x1_ind==x2_ind):
                        X=dataset.iloc[:, x1_ind - 1].values
                    else:
                        X = dataset.iloc[:, x1_ind - 1: x2_ind - 1].values
                    y = dataset.iloc[:, y1_dep - 1].values
                    X = np.array(X).reshape(-1, 1)
                    y = np.array(y).reshape(-1, 1)

                    from sklearn.preprocessing import PolynomialFeatures
                    poly_reg = PolynomialFeatures(degree=deg)
                    X_poly = poly_reg.fit_transform(X)
                    poly_reg.fit(X_poly, y)
                    lin_reg_2 = LinearRegression()
                    lin_reg_2.fit(X_poly, y)

                    text = form.cleaned_data['prediction']

                    y_pred1 = lin_reg_2.predict(poly_reg.fit_transform(text))

                    form = polynomial_regression()

                # a = X.tolist()
                # b = y.tolist()
                # l = []
                # l1 = []
                # for i in range(0, len(a)):
                #     l.append(a[i][0])
                # for i in range(0, len(b)):
                #     l1.append(b[i][0])

                X_grid = np.arange(min(X), max(X), 0.1)
                X_grid = X_grid.reshape((len(X_grid), 1))
                plt.scatter(X, y, color='red')
                plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
                plt.title('Polynomial Regression')
                plt.xlabel('Independent Variable')
                plt.ylabel('Dependent Variable')
                plt.savefig('mltool/static/csv/test.jpeg')
                plt.gcf().clear()

                args = {'form': form, 'result': y_pred1[0][0]}


                return render(request, self.template_name, args)


    class support_vector(TemplateView):
        template_name = 'C:\\Users\\ambuj\\Desktop\\mltool\\templates\\htmls\\svr.html'
        def get(self,request):
            form=mlr_regression()
            return render(request,self.template_name,{'form':form})

        def post(self,request):
            form=mlr_regression()
            if request.method == 'POST' or 'FILES' :

                form=IntForm(request.POST,request.FILES)
                if form.is_valid():


                    handle_uploaded_file(request.FILES['upload'])








                    x1_ind=form.cleaned_data['col_start']
                    x2_ind = form.cleaned_data['col_end']

                    y1_dep=form.cleaned_data['col_y']

                    file_name='C:\\Users\\ambuj\\Desktop\\mltool\\mltool\\static\\csv\\' + request.FILES['upload'].name

                    dataset = pd.read_csv(file_name)
                    if (x1_ind == x2_ind):
                        X = dataset.iloc[:, x1_ind - 1].values
                    else:
                        X = dataset.iloc[:, x1_ind - 1: x2_ind - 1].values
                    y = dataset.iloc[:, y1_dep - 1].values
                    X = np.array(X).reshape(-1, 1)
                    y = np.array(y).reshape(-1, 1)

                    from sklearn.svm import SVR
                    regressor = SVR(kernel='rbf')
                    regressor.fit(X, y)





                    text=form.cleaned_data['prediction']





                    y_pred1=regressor.predict(text)
                    form=mlr_regression()

                a = X.tolist()
                b=y.tolist()
                l=[]
                l1=[]
                for i in range(0,len(a)):
                    l.append(a[i][0])
                for i in range(0,len(b)):
                    l1.append(b[i][0])



                args = {'form': form, 'result': y_pred1[0],'x':l,'y':l1}


                return render(request, self.template_name, args)


    class decission_tree(TemplateView):

        template_name = 'C:\\Users\\ambuj\\Desktop\\mltool\\templates\\htmls\\dtr.html'
        def get(self,request):
            form = mlr_regression()
            return render(request,'htmls\\dtr_get.html',{'form':form})
        def post(self,request):
            form=mlr_regression()
            if request.method == 'POST' or 'FILES' :

                form=mlr_regression(request.POST,request.FILES)
                if form.is_valid():


                    handle_uploaded_file(request.FILES['upload'])








                    x1_ind=form.cleaned_data['col_start']
                    x2_ind = form.cleaned_data['col_end']

                    y1_dep=form.cleaned_data['col_y']

                    file_name='C:\\Users\\ambuj\\Desktop\\mltool\\mltool\\static\\csv\\' + request.FILES['upload'].name

                    dataset = pd.read_csv(file_name)
                    if (x1_ind == x2_ind):
                        X = dataset.iloc[:, x1_ind - 1].values
                    else:
                        X = dataset.iloc[:, x1_ind - 1: x2_ind - 1].values
                    y = dataset.iloc[:, y1_dep - 1].values
                    X = np.array(X).reshape(-1, 1)
                    y = np.array(y).reshape(-1, 1)

                    from sklearn.tree import DecisionTreeRegressor
                    regressor = DecisionTreeRegressor()
                    regressor.fit(X, y)





                    text=form.cleaned_data['prediction']





                    y_pred1=regressor.predict(text)
                    form=mlr_regression()

                a = X.tolist()
                b=y.tolist()
                l=[]
                l1=[]
                for i in range(0,len(a)):
                    l.append(a[i][0])
                for i in range(0,len(b)):
                    l1.append(b[i][0])

                X_grid = np.arange(min(X), max(X),0.1)
                X_grid = X_grid.reshape((len(X_grid), 1))
                plt.scatter(X, y, color='red')
                plt.plot(X_grid, regressor.predict(X_grid), color='blue')
                plt.title('Decission Tree Regression')
                plt.xlabel('Independent Variable')
                plt.ylabel('Dependent Variable')
                plt.savefig('mltool/static/csv/test.jpeg')
                plt.gcf().clear()

                args = {'form': form, 'result': y_pred1[0],'x':l,'y':l1}


                return render(request, self.template_name, args)
    class randon_forest(TemplateView):
        template_name = 'C:\\Users\\ambuj\\Desktop\\mltool\\templates\\htmls\\rfr.html'
        def get(self,request):
            form=mlr_regression()
            return render(request,'htmls\\rfr_get.html',{'form':form})

        def post(self, request):
            form = mlr_regression()
            if request.method == 'POST' or 'FILES':

                form = mlr_regression(request.POST, request.FILES)
                if form.is_valid():
                    handle_uploaded_file(request.FILES['upload'])

                    x1_ind = form.cleaned_data['col_start']
                    x2_ind = form.cleaned_data['col_end']

                    y1_dep = form.cleaned_data['col_y']

                    file_name = 'C:\\Users\\ambuj\\Desktop\\mltool\\mltool\\static\\csv\\' + request.FILES['upload'].name

                    dataset = pd.read_csv(file_name)
                    if(x1_ind==x2_ind):
                        X=dataset.iloc[:, x1_ind - 1].values
                    else:
                        X = dataset.iloc[:, x1_ind - 1: x2_ind - 1].values
                    y = dataset.iloc[:, y1_dep - 1].values
                    X = np.array(X).reshape(-1, 1)
                    y = np.array(y).reshape(-1, 1)


                    from sklearn.ensemble import RandomForestRegressor
                    regressor = RandomForestRegressor(n_estimators=300)
                    regressor.fit(X, y)

                    text = form.cleaned_data['prediction']

                    y_pred1 = regressor.predict(text)
                    form = mlr_regression()

                a = X.tolist()
                b = y.tolist()
                l = []
                l1 = []
                for i in range(0, len(a)):
                    l.append(a[i][0])
                for i in range(0, len(b)):
                    l1.append(b[i][0])

                X_grid = np.arange(min(X), max(X))
                X_grid = X_grid.reshape((len(X_grid), 1))
                plt.scatter(X, y, color='red')
                plt.plot(X_grid, regressor.predict(X_grid), color='blue')
                plt.title('Random Forest Regression')
                plt.xlabel('Independent Variable')
                plt.ylabel('Dependent Variable')
                plt.savefig('mltool/static/csv/test.jpeg')
                plt.gcf().clear()

                args = {'form': form, 'result': y_pred1[0], 'x': l, 'y': l1}

                return render(request, self.template_name, args)

except :
    def homepage(request):
        return render(request,'htmls\\documentation')

















