"""mltool URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static

from django.urls import path
from django.views.generic import TemplateView
from mltool import views as mltool_views
from django.conf.urls import handler404, handler500
from . import  views
from mltool.views import linear,multiple,support_vector,polynomial,randon_forest,decission_tree,documentation
urlpatterns = [
    path('index.html', views.index),
    path('mlr.html',multiple.as_view()),
    path('svr.html',support_vector.as_view()),
    path('pr.html',polynomial.as_view()),
    path('dtr.html',decission_tree.as_view()),
    path('rfr.html',randon_forest.as_view()),
    path('documentation.html',views.documentation),



    path('admin/', (admin.site.urls),name=admin),
    path('lr.html',linear.as_view()),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


handler404 = mltool_views.error_404
handler500 = mltool_views.error_500
