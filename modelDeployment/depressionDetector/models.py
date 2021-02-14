from django.db import models


class twitter_id(models.Model):
    search = models.CharField(max_length=100)
    tweets_no = models.PositiveIntegerField(default=0)
    created = models.DateTimeField(auto_now=True)

    def __str__(self):
        return '{}'.format(self.search)

    class Meta:
        verbose_name_plural = 'Twitter ID'