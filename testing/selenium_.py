from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import time


driver = webdriver.Chrome(
        executable_path='../driver/chromedriver.exe'
    )

driver.get("https://www.onlinetours.ru/")

time.sleep(3)

element = driver.find_element_by_class_name('search-form__field')

element2 = element.find_element_by_tag_name('input')
element2.send_keys('Крым')

#elem_find = driver.find_element_by_class_name("search-form__submit-button-state")
elem_find = driver.find_element_by_xpath('/html/body/div[2]/article/div[1]/div/div[2]/div/div[1]/div/div[1]/div/div[5]/a/div[2]')
elem_find.click()
time.sleep(2)
elem_days = driver.find_element_by_xpath('/html/body/div[2]/article/div[1]/div/div[2]/div/div[1]/div/div[2]/div[2]/div/div[1]/div[1]/div[2]/div[4]/div')
elem_days.click()
time.sleep(2)
elem_days2 = driver.find_element_by_xpath('/html/body/div[2]/article/div[1]/div/div[2]/div/div[1]/div/div[2]/div[2]/div/div[1]/div[1]/div[2]/div[8]/div')
elem_days2.click()
time.sleep(2)
day1 = '/html/body/div[2]/article/div[1]/div/div[2]/div/div[1]/div/div[2]/div[2]/div/div[1]/div[2]/div[2]/div[3]/div/div[2]/div[5]/div[2]/div'
day2 = '/html/body/div[2]/article/div[1]/div/div[2]/div/div[1]/div/div[2]/div[2]/div/div[1]/div[2]/div[2]/div[3]/div/div[3]/div[4]/div[3]/div'

#elem_day = driver.find_element_by_xpath(day1)
#elem_day.click()

elem_day2 = driver.find_element_by_xpath(day2)
elem_day2.click()

time.sleep(2)
elem_find.click()

#time.sleep(1)
#elem_find.click()





#driver.close()
def test_title():
    if driver.title == 'OK.RU':
        print('Title test passed.')
    else:
        print('Actual title is {}'.format(driver.title))


'''
def test_user_login():
    login_field = driver.find_element_by_id("field_email")
    login_field.send_keys("89102272132")
    password_field = driver.find_element_by_id("field_password")
    password_field.send_keys("testpass78")
    driver.find_element_by_xpath("//input[@value='Войти']").click()
    true_login = driver.find_element_by_xpath('//*[@id="hook_Block_Navigation"]/div/div/a[1]/span').text
    if true_login == 'test test':
        print('Authorization test passed.')


def test_like():
    driver.get('https://ok.ru/profile/515292222790')

    like_counts = driver.find_elements_by_class_name('widget-list_i')[1].find_element_by_class_name('widget_count').text
    driver.find_elements_by_class_name('widget-list_i')[1].find_element_by_class_name('widget_cnt').click()
    time.sleep(1)
    like_counts_after = driver.find_elements_by_class_name('widget-list_i')[1].find_element_by_class_name('widget_count').text
    if abs(int(like_counts) - int(like_counts_after)):
        print('Like test passed. {}, {}'.format(like_counts, like_counts_after))
    else:
        print(like_counts, like_counts_after)


def test_lang():
    driver.get("https://www.ok.ru/")
    field_before = driver.find_element_by_xpath('//*[@id="hook_Block_HeaderTopFriendsInToolbar"]/a/div/div').text
    driver.find_element_by_xpath('//*[@id="mailRuToolbar"]/table/tbody/tr/td[2]/div/span').click()
    driver.find_element_by_xpath('//*[@id="mailRuToolbar"]/table/tbody/tr/td[2]/div/div/div/div/a[1]').click()
    time.sleep(10)
    field_after = driver.find_element_by_xpath('//*[@id="hook_Block_HeaderTopFriendsInToolbar"]/a/div/div').text

    if field_after == 'Friends':
        print('Lang test passed.')


def test_posting():
    driver.find_element_by_class_name('pf-with-ava_cnt').click()
    test_txt = 'test str'
    time.sleep(5)
    driver.find_element_by_class_name('posting_itx').send_keys(test_txt)
    driver.find_element_by_class_name('posting_submit').click()
    time.sleep(2)
    last_post_text = driver.find_elements_by_class_name('media-text_cnt_tx')[0].text
    if last_post_text == test_txt:
        print('Post test passed')
    else:
        print(last_post_text)


def test_deleting():
    all_feeds = driver.find_elements_by_class_name('feed-w')
    if check_el_by_class(all_feeds[0]):
        last_feed = all_feeds[1]
    else:
        last_feed = all_feeds[0]

    txt = last_feed.find_element_by_class_name('media-text_cnt_tx').text
    last_feed.find_element_by_class_name('media-text_a').click()
    time.sleep(10)
    driver.find_element_by_class_name('mlr_top_ac').click()
    time.sleep(1)
    driver.find_element_by_link_text('Удалить').click()
    time.sleep(2)
    # driver.find_elements_by_class_name('u-menu_li')[2].click()
    # driver.find_elements_by_class_name('u-menu_li')[2].find_element_by_class_name('u-menu_a').click()
    driver.get("https://www.ok.ru/")

    posts = driver.find_elements_by_class_name('media-text_cnt_tx')

    if len(posts) == 0:
        print('Del test passed')
        return
    last_post_text = posts[0].text
    if txt != last_post_text:
        print('Del test passed')


def check_el_by_class(subtree):
    try:
        subtree.find_element_by_class_name('__no-ava')
        return True
    except NoSuchElementException:
        return False


# test_title()
test_user_login()
# test_like()
# test_lang()

test_posting()
test_deleting()

'''