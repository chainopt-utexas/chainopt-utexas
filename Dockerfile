FROM chainopt/models-api:latest as build
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org scikit-learn==0.21.2 xgboost==0.90 requests
COPY ./projects /app/models

FROM chainopt/models-api:test as test
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org scikit-learn==0.21.2 xgboost==0.90 requests
COPY ./projects /app/models

FROM build as prod