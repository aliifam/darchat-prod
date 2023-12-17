# Chatbot Darurat

## build image

```
docker build -t gcr.io/capstone-proj-406511/darchat:v1 .
```

## run container from image

```
docker run -d -p 8080:3000 --name chatbot gcr.io/capstone-proj-406511/darchat:v1
```

## push image to gcr

```
docker push gcr.io/capstone-proj-406511/darchat:v1
```

## deploy to cloud run

```
gcloud run deploy --image gcr.io/capstone-proj-406511/darchat:v1 --platform managed
```
