from django.conf.urls import include, url
from django.contrib import admin
# from bi_project import views
urlpatterns = [
    # Examples:
    # url(r'^$', 'bi_project.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
    url(r'^$', 'bi_medical.views.home', name='home'),

]
