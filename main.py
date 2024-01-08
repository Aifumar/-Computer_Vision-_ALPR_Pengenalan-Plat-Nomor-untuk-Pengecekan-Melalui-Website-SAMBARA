import time
import easyocr
import cv2
import numpy as np
import requests
import re
from io import BytesIO
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ultralytics import YOLO
    

url = 'https://sambara.bapenda.jabarprov.go.id/sambara_lite_plopd/'
url_gagal = 'https://sambara.bapenda.jabarprov.go.id/sambara_lite_plopd/landing'

mode_turbo = 0 #-------> Atur mode (1 = untuk mode turbo,  0 = untuk mode normal(Akan menampilkan web sambara)) <---------

# Inisialisasi waktu awal dan deteksi plat
w_mulai=time.time()
model = YOLO('best.pt')
desired_width = 1080
img_=cv2.imread('image/img1.jpg') # Pemilihan gambar untuk dieksekusi
original_height, original_width, _ = img_.shape
aspect_ratio = desired_width / original_width
desired_height = int(original_height * aspect_ratio)
img=cv2.resize(img_, (desired_width, desired_height))
results = model.track(img, persist=True)
results_ = results[0].plot()

# Pembuatan Bounding Box pada plat yang terdeteksi & Cropping
for result in results:
    cap_count=0
    boxes = result.boxes.cpu().numpy()
    print('Posisi Bounding Box :')
    for box in boxes:
        r = box.xyxy[0].astype(int)
        x1, y1, x2, y2 = r
        print(r)
        cropped_image = img[y1:y2, x1:x2]
        
        # Hitam Putih
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        
        # Noise Reduction
        nored = cv2.bilateralFilter(gray_image, 100, 50, 50)
        
        # Metode Dilatasi
        kernel = np.ones((2, 2), np.uint8)
        dilatasi = cv2.dilate(nored, kernel, iterations=1)
        
        # Thresholding
        _, thresholded_image = cv2.threshold(dilatasi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
        # EasyOCR
        reader = easyocr.Reader(lang_list=['id'])
        image = np.copy(cropped_image)
        image_ = np.copy(cropped_image)
        results = reader.readtext(image, allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        threshold = 0.25
        for t_, t in enumerate(results):
            print(t)
            bbox, text, score = t
            if score > threshold:
                cv2.rectangle(image_, bbox[0], bbox[2], (0, 0, 0), 5)
                cv2.putText(image_, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 255, 255), 2)
        extracted_text = [result[1] for result in results]
        print('\nHasil OCR :')
        ocr=' '.join(extracted_text)
        s=''.join(extracted_text)
        if s[0] == '0':
            s = 'D' + s[1:]
            ocr = 'D' + ocr[1:]
        elif s[0] == '8':
            s = 'B' + s[1:]
            ocr = 'B' + ocr[1:]
        elif s[0] == '2':
            s = 'Z' + s[1:]
            ocr = 'Z' + ocr[1:]
        print(ocr)
        plat=re.split('(\d+)',s) # Library RegEx untuk split karakter plat nomor
        
        # Inisialisasi driver web Selenium
        if mode_turbo:
            options = webdriver.ChromeOptions()
            options.add_argument('--headless')
            driver = webdriver.Chrome(options=options)
            driver.get(url)
        else:
            driver = webdriver.Chrome()
            driver.get(url)

        while True:
            if mode_turbo:
                wait = WebDriverWait(driver, 10)
                wait.until(EC.presence_of_element_located((By.NAME, "nopol1")))
                
                # Inspeksi Elemen dengan Selenium + isi form
                driver.find_element(By.NAME, "nopol1").send_keys(plat[0])
                driver.find_element(By.NAME, "nopol2").send_keys(plat[1])
                driver.find_element(By.NAME, "nopol3").send_keys(plat[2])
                image_element = driver.find_element(By.XPATH, "/html/body/div/div/div/div/div[2]/div[2]/form/table[3]/tbody/tr[1]/td/p/img")
                image_url = image_element.get_attribute('src')
                print('\nLink Gambar Captcha :')
                print(image_url)
                response = requests.get(image_url)
                
                # Library BytesIO untuk mengambil gambar captcha dari elemen web
                image_data = BytesIO(response.content)
                
                # Pengolahan citra untuk bypass captcha
                img_gray = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                _, thresholded_image = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)
                reader = easyocr.Reader(lang_list=['id'])
                result = reader.readtext(thresholded_image, allowlist = '0123456789')
                result_sorted = sorted(result, key=lambda x: x[0][0][0])
                merged_text = ''.join([detection[1] for detection in result_sorted])
                print('\nHasil Tebakan Captcha :')
                print(merged_text)
                driver.find_element(By.XPATH, '//*[@id="secure"]').send_keys(merged_text)
                driver.find_element(By.XPATH, '/html/body/div/div/div/div/div[2]/div[2]/form/table[4]/tbody/tr/th/button').click()
                
                # Inisialisasi kegagalan bypass captcha
                if driver.current_url != url_gagal:
                    wait.until(EC.presence_of_element_located((By.XPATH, '/html/body/div')))
                    break
                cap_count += 1
                print('Tebakan Captcha Gagal, Mencoba Ulang...')

            else:
                wait = WebDriverWait(driver, 10)
                time.sleep(2)
                
                # Inspeksi Elemen dengan Selenium + isi form
                driver.find_element(By.NAME, "nopol1").send_keys(plat[0])
                driver.find_element(By.NAME, "nopol2").send_keys(plat[1])
                driver.find_element(By.NAME, "nopol3").send_keys(plat[2])
                image_element = driver.find_element(By.XPATH, "/html/body/div/div/div/div/div[2]/div[2]/form/table[3]/tbody/tr[1]/td/p/img")
                image_url = image_element.get_attribute('src')
                print('\nLink Gambar Captcha :')
                print(image_url)
                response = requests.get(image_url)
                
                # Library BytesIO untuk mengambil gambar captcha dari elemen web
                image_data = BytesIO(response.content)
                
                # Pengolahan citra untuk bypass captcha
                img_gray = cv2.imdecode(np.frombuffer(image_data.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                _, thresholded_image2 = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)
                reader = easyocr.Reader(lang_list=['id'])
                result = reader.readtext(thresholded_image2, allowlist = '0123456789')
                result_sorted = sorted(result, key=lambda x: x[0][0][0])
                merged_text = ''.join([detection[1] for detection in result_sorted])
                print('\nHasil Tebakan Captcha :')
                print(merged_text)
                driver.find_element(By.XPATH, '//*[@id="secure"]').send_keys(merged_text)
                driver.find_element(By.XPATH, '/html/body/div/div/div/div/div[2]/div[2]/form/table[4]/tbody/tr/th/button').click()
                
                # Inisialisasi kegagalan bypass captcha
                if driver.current_url != url_gagal:
                    time.sleep(2)
                    break
                cap_count += 1
                print('Tebakan Captcha Gagal, Mencoba Ulang...')

        print(f'\nJumlah Tebakan Captcha Gagal : {cap_count}')
        
        # Web Scrapping dengan selenium
        merk = driver.find_element(By.XPATH, '/html/body/div/div/div[1]/div/div[2]/div[2]/div/table[1]/tbody/tr[3]/td[3]').text
        model = driver.find_element(By.XPATH, '/html/body/div/div/div[1]/div/div[2]/div[2]/div/table[1]/tbody/tr[4]/td[3]').text
        thn = driver.find_element(By.XPATH, '/html/body/div/div/div[1]/div/div[2]/div[2]/div/table[1]/tbody/tr[5]/td[3]').text
        warna = driver.find_element(By.XPATH, '/html/body/div/div/div[1]/div/div[2]/div[2]/div/table[1]/tbody/tr[6]/td[3]').text
        no_rang = driver.find_element(By.XPATH, '/html/body/div/div/div[1]/div/div[2]/div[2]/div/table[1]/tbody/tr[7]/td[3]').text
        no_mes = driver.find_element(By.XPATH, '/html/body/div/div/div[1]/div/div[2]/div[2]/div/table[1]/tbody/tr[8]/td[3]').text
        pjk = len(driver.find_element(By.XPATH, '/html/body/div/div/div[1]/div/div[2]/div[2]/div/table[2]/tbody/tr[14]/td[3]').text)
        tgl_pjk = driver.find_element(By.XPATH, '/html/body/div/div/div[1]/div/div[2]/div[2]/div/table[2]/tbody/tr[12]/td[3]').text
        
        # Menampilkan hasil 
        print('\n========== DETAIL KENDARAAN BERMOTOR ==========')
        print(f'Merk : {merk}\nModel : {model}\nTahun : {thn}\nWarna : {warna}\nNo Rangka : {no_rang}\nNo Mesin : {no_mes}')
        if pjk > 20:
            print(f'Pajak : MATI ({tgl_pjk})')
        else:
            print(f'Pajak : HIDUP ({tgl_pjk})')
        print('===============================================\n')
        
        if mode_turbo:
            driver.quit()
        w_selesai=time.time()
        print(f'Waktu Total Cycle : {w_selesai - w_mulai} detik')        
        
        # Menampilkan Semua Gambar (Tidak akan tereksekusi apabila terjadi error pada laman web)
        cv2.imshow('Predict Image', results_)
        cv2.imshow('Potongan Gambar', cropped_image)
        cv2.imshow('Hitam Putih', gray_image)
        cv2.imshow('Noise Reduction', nored)
        cv2.imshow('Dilatasi', dilatasi)
        cv2.imshow('Threshold', thresholded_image)
        cv2.imshow('Deteksi Kata',image_)
        cv2.imwrite('OutImage/Predict_Image.jpg', results_)
        cv2.imwrite('OutImage/Potongan_Gambar.jpg', cropped_image)
        cv2.imwrite('OutImage/Hitam_Putih.jpg', gray_image)
        cv2.imwrite('OutImage/Noise_Reduction.jpg', nored)
        cv2.imwrite('OutImage/Dilatasi.jpg', dilatasi)
        cv2.imwrite('OutImage/Threshold.jpg', thresholded_image)
        cv2.imwrite('OutImage/Deteksi Kata.jpg', image_)
        if not mode_turbo:
            cv2.imshow('Binary', thresholded_image2)
        cv2.waitKey(0)
