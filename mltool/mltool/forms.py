from django import forms

class IntForm(forms.Form):
    #your_name = forms.CharField(label='Your name', max_length=100)

    prediction=forms.IntegerField(min_value=0)
    col_start=forms.IntegerField(min_value=1)
    #col_end=forms.IntegerField()
    col_y=forms.IntegerField()
    upload = forms.FileField()


class mlr_regression(forms.Form):
    prediction = forms.IntegerField(min_value=0)
    col_start = forms.IntegerField(min_value=1)
    col_end=forms.IntegerField()
    col_y = forms.IntegerField()
    upload = forms.FileField()

class polynomial_regression(forms.Form):
    prediction = forms.IntegerField(min_value=0)
    col_start = forms.IntegerField(min_value=1)
    col_end = forms.IntegerField()
    col_y = forms.IntegerField()
    degree = forms.IntegerField(min_value=0)
    upload = forms.FileField()




    '''def clean(self):
        cleaned_data = super(IntForm, self).clean()
        prediction = cleaned_data.get('prediction')
        upload=cleaned_data.get('upload')
        if not prediction and not upload:
            raise forms.ValidationError('You have to write something!')'''
