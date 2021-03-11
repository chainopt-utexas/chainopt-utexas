FROM chainopt/models-api:2.0 as build
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org scikit-learn==0.21.2 xgboost==0.90
COPY ./models /app/models

FROM chainopt/models-api:test-2.0 as test
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org scikit-learn==0.21.2 xgboost==0.90
COPY ./models /app/models

FROM build as prod
