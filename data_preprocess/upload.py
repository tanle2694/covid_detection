from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from apiclient.http import MediaFileUpload
import os
import argparse

# Access this link
# https://developers.google.com/drive/api/v3/quickstart/python

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']

def main(folder_image, folder_id, output_id_file):
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('drive', 'v3', credentials=creds)
    nb_file = 0
    f_write = open(output_id_file, "w")
    for sub_dir, _, fs in os.walk(folder_image):
        for f in fs:
            nb_file += 1
            file_name = os.path.join(sub_dir, f)
            print("Uploaded: {} file ".format(nb_file))
            print("Current upload: {}".format(file_name))
            file_metadata = {
                               'name': f,
                                'parents': [ folder_id ]
                            }
            media = MediaFileUpload(file_name,
                mimetype='image/png')
            file = service.files().create(body=file_metadata,
                media_body=media,
                fields='id').execute()
            f_write.write("{}\t{}\n".format(file_name , file.get('id')))
    f_write.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_image", type=str, default="")
    parser.add_argument("--folder_id", type=str, default="")
    parser.add_argument("--output_id_file", type=str, default="")
    args = parser.parse_args()
    main(args.folder_image, args.folder_id, args.output_id_file)