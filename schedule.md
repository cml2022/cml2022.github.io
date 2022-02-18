---
layout: default
title: Schedule
---

Links to the course material will be provided in the schedule below
after each class.

## The course schedule

<table rules="groups" style="width:100%;border-collapse: collapse;">
  <thead style="border-bottom: 1px solid #000;">
    <tr>
      <th style="text-align:left;" width="10%">Week</th>
      <th style="text-align:left;" width="30%">Monday</th>
      <th style="text-align:left;" width="30%">Wednesday</th>
    </tr>
  </thead>
  <tbody style="border-bottom: 1px solid #000;">
{% for week in site.data.schedule %}
    <tr style="border-bottom: 1px solid #000;">
    <td style="text-align:left;"> {{ week.week }} </td>
    {% for date in week.dates %}
            <td valign="top"> <span style="color:gray">{{date[0]}}</span><br/>
                {% if date[1].topic %}
                    {{ date[1].topic }}&nbsp;
                {% else %}
                    <em style="color: red">{{date[1]}}</em>
                {% endif %}
                {% if date[1].links %}
                    <br/>
                    [{% for link in date[1].links %}<a href="{{ link[1] }}">{{ link[0] }}</a>{% unless forloop.last %}, {% endunless %}{% endfor %}]
                {% endif %}
                {% if date[1].reading %}
                    <br/>
                    Reading: {{ date[1].reading }}
                {% endif %}
            </td>
    {% endfor %}
    </tr>
{% endfor %}
  </tbody>
</table>
