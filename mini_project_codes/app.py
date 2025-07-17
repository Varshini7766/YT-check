from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from textblob import TextBlob
import base64
import io
from concurrent.futures import ThreadPoolExecutor
from functools import partial

app = Flask(__name__)

# Set your YouTube API key
api_key = "AIzaSyCsCwBsv11xO8BR66mi-iw5WSDhH70-d28"

# Create a YouTube API client
youtube = build('youtube', 'v3', developerKey=api_key)

def get_video_details(video_id):
    try:
        response = youtube.videos().list(
            part='snippet',
            id=video_id,
            fields='items(snippet(title))'
        ).execute()

        if response['items']:
            return response['items'][0]['snippet']['title']
        return "Video title not found"
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error fetching video title"

def get_comments_page(youtube, video_id, page_token=None):
    try:
        return youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText',
            maxResults=100,
            pageToken=page_token,
            fields='items(snippet(topLevelComment(snippet(textDisplay)))),nextPageToken'
        ).execute()
    except Exception as e:
        print(f"Error fetching comments: {e}")
        return None

def get_video_comments(video_id, max_comments=500):
    comments = []
    next_page_token = None
    
    while len(comments) < max_comments:
        try:
            response = get_comments_page(youtube, video_id, next_page_token)
            if not response:
                break

            page_comments = [
                item['snippet']['topLevelComment']['snippet']['textDisplay']
                for item in response.get('items', [])
            ]
            comments.extend(page_comments)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        except Exception as e:
            print(f"An error occurred: {e}")
            break

    return comments[:max_comments]

def analyze_sentiment_batch(comments, batch_size=100):
    sentiments = []
    
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        with ThreadPoolExecutor() as executor:
            batch_sentiments = list(executor.map(analyze_sentiment, batch))
            sentiments.extend(batch_sentiments)
    
    return sentiments

def analyze_sentiment(comment):
    try:
        analysis = TextBlob(comment)
        sentiment = analysis.sentiment() if callable(analysis.sentiment) else analysis.sentiment
        # If sentiment is a tuple or object, extract polarity safely
        polarity = getattr(sentiment, 'polarity', None)
        if polarity is None and isinstance(sentiment, tuple):
            polarity = sentiment[0]
        if polarity is None:
            print(f"Sentiment object has no polarity: {sentiment}")
            return 'Neutral'
        if polarity > 0:
            return 'Positive'
        elif polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'
    except Exception as e:
        print(f"Error in analyze_sentiment: {e}")
        return 'Neutral'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video_sentiment():
    video_id = request.form['video_id']
    
    with ThreadPoolExecutor() as executor:
        future_title = executor.submit(get_video_details, video_id)
        future_comments = executor.submit(get_video_comments, video_id)
        
        video_title = future_title.result()
        comments = future_comments.result()

    if not comments:
        return jsonify({"error": "No comments found or error occurred."}), 400

    sentiments = analyze_sentiment_batch(comments)
    
    df = pd.DataFrame({
        'Comment': comments,
        'Sentiment': sentiments
    })

    sentiment_counts = df['Sentiment'].value_counts()
    sentiment_percentages = sentiment_counts / len(df) * 100

    # Updated colors for both charts
    colors = ['#4CAF50', '#FFC107', '#F44336']  # Green, Yellow, Red

    plt.figure(figsize=(8, 6), dpi=100)
    plt.pie(sentiment_percentages, labels=sentiment_percentages.index, autopct='%1.1f%%', 
            startangle=90, colors=colors)
    plt.title('Sentiment Distribution')
    
    pie_buffer = io.BytesIO()
    plt.savefig(pie_buffer, format='png', bbox_inches='tight')
    pie_buffer.seek(0)
    pie_chart = base64.b64encode(pie_buffer.getvalue()).decode()
    plt.close()

    plt.figure(figsize=(8, 6), dpi=100)
    plt.bar(range(len(sentiment_counts)), list(sentiment_counts.values), color=colors)
    plt.xticks(range(len(sentiment_counts)), list(map(str, sentiment_counts.index)))
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    for i, v in enumerate(sentiment_counts.values):
        plt.text(i, v, str(int(v)), ha='center', va='bottom')
    
    bar_buffer = io.BytesIO()
    plt.savefig(bar_buffer, format='png', bbox_inches='tight')
    bar_buffer.seek(0)
    bar_chart = base64.b64encode(bar_buffer.getvalue()).decode()
    plt.close()

    results = {
        "video_title": video_title,
        "sentiment_percentages": sentiment_percentages.to_dict(),
        "pie_chart": pie_chart,
        "bar_chart": bar_chart
    }

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)